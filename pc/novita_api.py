"""
Novita.ai API Client

Provides Python interface for interacting with Novita.ai GPU management API.
Phase 1: Basic connectivity and user info retrieval.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Dict, Optional, Any


class NovitaAPIError(Exception):
    """Base exception for Novita API errors."""
    pass


class NovitaAuthenticationError(NovitaAPIError):
    """Exception raised for authentication failures."""
    pass


class NovitaAPIClient:
    """Client for interacting with Novita.ai API."""
    
    BASE_URL = "https://api.novita.ai/v1"
    GPU_INSTANCE_BASE_URL = "https://api.novita.ai/gpu-instance/openapi/v1"
    DEFAULT_TIMEOUT = 15  # seconds
    
    def __init__(self, api_key: str):
        """
        Initialize Novita API client.
        
        Args:
            api_key (str): Novita.ai API key for authentication.
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        self.logger = logging.getLogger(__name__)
        
        # Configure retry strategy for resilience
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _parse_ports_string(self, ports_str: str) -> list:
        """
        Parse port mappings string into API-expected format.
        
        Args:
            ports_str: Port mappings string (e.g., "8080/http, 6006/tcp").
        
        Returns:
            list: List of port mapping dicts in API format.
                  Format: [{"port": 8080, "protocol": "http"}, ...]
        """
        if not ports_str or not ports_str.strip():
            return []
        
        parsed_ports = []
        # Split by comma and process each port mapping
        for port_spec in ports_str.split(','):
            port_spec = port_spec.strip()
            if not port_spec:
                continue
            
            try:
                # Expected format: "port/protocol" or just "port"
                if '/' in port_spec:
                    port_str, protocol = port_spec.split('/', 1)
                    port = int(port_str.strip())
                    protocol = protocol.strip().lower()
                else:
                    # Default to tcp if no protocol specified
                    port = int(port_spec.strip())
                    protocol = "tcp"
                
                # Validate protocol
                if protocol not in ['tcp', 'http', 'https', 'udp']:
                    self.logger.warning(
                        f"Unknown protocol '{protocol}' for port {port}, defaulting to tcp"
                    )
                    protocol = "tcp"
                
                parsed_ports.append({
                    "port": port,
                    "protocol": protocol
                })
                
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Failed to parse port spec '{port_spec}': {e}")
                continue
        
        return parsed_ports
    
    def _extract_list(self, data: Dict[str, Any], candidates: list, context: str = "data") -> list:
        """
        Extract a list from API response data by trying multiple possible keys.
        
        Args:
            data: The API response data dictionary.
            candidates: List of possible key names to try in order.
            context: Context string for logging (e.g., "products", "clusters").
        
        Returns:
            list: The extracted list, or empty list if none found.
        """
        if not isinstance(data, dict):
            self.logger.warning(f"Expected dict for {context}, got {type(data).__name__}")
            return []
        
        # Try each candidate key in order
        for key in candidates:
            # Support nested keys like "data.items"
            if "." in key:
                parts = key.split(".")
                current = data
                try:
                    for part in parts:
                        current = current.get(part, {})
                        if current is None:
                            break
                    if current and isinstance(current, list):
                        self.logger.debug(f"Found {context} at key: {key}")
                        return current
                except (AttributeError, TypeError):
                    continue
            else:
                # Simple key lookup
                value = data.get(key)
                if value is not None and isinstance(value, list):
                    self.logger.debug(f"Found {context} at key: {key}")
                    return value
        
        # None of the candidate keys worked
        self.logger.warning(
            f"Could not extract {context} from response. "
            f"Tried keys: {candidates}. Available keys: {list(data.keys())}"
        )
        return []
    
    def _redact_sensitive_data(self, text: str) -> str:
        """
        Redact sensitive information from text (API keys, tokens).
        
        Args:
            text (str): Text that may contain sensitive information.
            
        Returns:
            str: Text with sensitive information redacted.
        """
        if not text:
            return text
        
        # Redact anything that looks like a bearer token or API key
        import re
        # Redact Authorization headers
        text = re.sub(r'(Authorization["\s:]+Bearer\s+)[^\s"\']+', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        # Redact api_key fields in JSON
        text = re.sub(r'("api_key"\s*:\s*")[^"]+', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        # Redact bearer tokens in general
        text = re.sub(r'(Bearer\s+)[A-Za-z0-9_\-\.]+', r'\1[REDACTED]', text, flags=re.IGNORECASE)
        
        return text
    
    def _log_response_error(self, response: requests.Response, context: str) -> None:
        """
        Log detailed error information from a response.
        
        Args:
            response: The response object from requests.
            context: Context string describing what operation failed.
        """
        try:
            # Get response body, truncate if too long
            body = response.text[:500] if response.text else "(empty)"
            if len(response.text) > 500:
                body += "... (truncated)"
            
            # Redact sensitive information
            body = self._redact_sensitive_data(body)
            
            self.logger.error(
                f"{context} - Status: {response.status_code}, "
                f"URL: {response.url}, Body: {body}"
            )
        except Exception as e:
            self.logger.error(f"{context} - Could not log response details: {e}")
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information and credit balance.
        Tries multiple endpoints to find working user info endpoint.
        
        Returns:
            dict: User information including credits/balance.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        # Try multiple possible endpoints for user info
        endpoints_to_try = [
            f"{self.GPU_INSTANCE_BASE_URL}/user",  # Try GPU instance API path first
            f"{self.BASE_URL}/user",  # Original endpoint
            f"{self.BASE_URL}/account/info",  # Alternative path
        ]
        
        last_error = None
        
        for endpoint in endpoints_to_try:
            try:
                response = self.session.get(
                    endpoint,
                    timeout=self.DEFAULT_TIMEOUT
                )
                
                if response.status_code == 401:
                    self._log_response_error(response, "Authentication failed")
                    raise NovitaAuthenticationError("Invalid API key")
                
                if response.status_code == 404:
                    # This endpoint doesn't exist, try next one
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                self.logger.info(f"Successfully retrieved user info from {endpoint}")
                return data
                
            except NovitaAuthenticationError:
                # Re-raise authentication errors immediately
                raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    # Try next endpoint
                    last_error = e
                    continue
                else:
                    # Other HTTP error, log and raise
                    if e.response is not None:
                        self._log_response_error(e.response, f"Failed to get user info from {endpoint}")
                        last_error = NovitaAPIError(
                            f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                        )
                    else:
                        self.logger.error(f"Failed to get user info - No response available")
                        last_error = NovitaAPIError(f"API request failed: {e}")
                    break
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to get user info from {endpoint} - Request exception: {e}")
                last_error = NovitaAPIError(f"API request failed: {e}")
                break
        
        # If we got here, all endpoints failed
        if last_error:
            raise last_error
        else:
            raise NovitaAPIError("No working user info endpoint found (all returned 404)")
    
    def list_gpu_products(self) -> list:
        """
        List available GPU products.
        
        Returns:
            list: Available GPU products with fields like id, name, cpuPerGpu, memoryPerGpu, 
                  gpuNum, price, regions, availableDeploy.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/products"
        try:
            response = self.session.get(
                endpoint,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info("Successfully retrieved GPU products")
            
            # Extract products list from multiple possible keys
            products = self._extract_list(
                data,
                ["data.items", "items", "products", "data.products", "data"],
                context="products"
            )
            return products
            
        except requests.exceptions.HTTPError as e:
            # Log structured error details
            if e.response is not None:
                self._log_response_error(e.response, "Failed to list GPU products")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to list GPU products - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list GPU products - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def list_clusters(self) -> list:
        """
        List available clusters.
        
        Returns:
            list: Available clusters with fields like id, name, supportedGpuTypes.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/clusters"
        try:
            response = self.session.get(
                endpoint,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info("Successfully retrieved clusters")
            
            # Extract clusters list from multiple possible keys
            clusters = self._extract_list(
                data,
                ["data.items", "items", "clusters", "data.clusters", "data"],
                context="clusters"
            )
            return clusters
            
        except requests.exceptions.HTTPError as e:
            # Log structured error details
            if e.response is not None:
                self._log_response_error(e.response, "Failed to list clusters")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to list clusters - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list clusters - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def list_instances(self, page_size: int = 100, page_num: int = 1) -> list:
        """
        List GPU instances.
        
        Args:
            page_size (int): Number of instances per page (default: 100).
            page_num (int): Page number (default: 1).
        
        Returns:
            list: List of instances with fields like id, name, status, clusterId, clusterName,
                  productId, productName, imageUrl, cpuNum, memory, gpuNum, portMappings,
                  billingMode, spotStatus, createdAt.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instances"
        try:
            params = {
                "pageSize": page_size,
                "pageNum": page_num
            }
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract instances list from multiple possible keys
            instances = self._extract_list(
                data,
                ["data.items", "items", "instances", "data.instances", "data"],
                context="instances"
            )
            self.logger.info(f"Successfully retrieved {len(instances)} instances")
            return instances
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, "Failed to list instances")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to list instances - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list instances - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific instance.
        
        Args:
            instance_id (str): Instance ID to retrieve.
        
        Returns:
            dict: Instance details including connection info, logs, and status.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instance"
        try:
            params = {"instanceId": instance_id}
            response = self.session.get(
                endpoint,
                params=params,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully retrieved instance {instance_id}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, f"Failed to get instance {instance_id}")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to get instance - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get instance - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def create_instance(self, product_id: str, image_url: str, name: str = None, 
                       cluster_id: str = None, ports: str = None, envs: list = None, 
                       command: str = None) -> Dict[str, Any]:
        """
        Create a new GPU instance.
        
        Args:
            product_id (str): GPU product ID.
            image_url (str): Docker image URL.
            name (str, optional): Instance name (auto-generated if not provided).
            cluster_id (str, optional): Cluster ID (API will auto-select if not provided).
            ports (str, optional): Port mappings string (e.g., "8080/http, 6006/tcp") or 
                                   pre-formatted list of port dicts.
            envs (list, optional): List of environment variable dicts with 'key' and 'value'.
            command (str, optional): Container command override.
        
        Returns:
            dict: Created instance data including the new instance ID.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instance/create"
        try:
            payload = {
                "productId": product_id,
                "imageUrl": image_url,
                "kind": "gpu"  # Required by API
            }
            
            if name:
                payload["name"] = name
            if cluster_id:
                payload["clusterId"] = cluster_id
            
            # Transform ports to expected API format
            if ports:
                if isinstance(ports, str):
                    # Parse string format "8080/http, 6006/tcp" to list of dicts
                    parsed_ports = self._parse_ports_string(ports)
                    if parsed_ports:
                        payload["ports"] = parsed_ports
                elif isinstance(ports, list):
                    # Already in list format, use as-is
                    payload["ports"] = ports
                else:
                    self.logger.warning(f"Unexpected ports type: {type(ports)}, skipping")
            
            # Transform envs to expected API format
            if envs:
                if isinstance(envs, list):
                    # Validate that envs are dicts with 'key' and 'value'
                    validated_envs = []
                    for env in envs:
                        if isinstance(env, dict) and 'key' in env and 'value' in env:
                            validated_envs.append(env)
                        else:
                            self.logger.warning(f"Invalid env format: {env}, expected dict with 'key' and 'value'")
                    if validated_envs:
                        payload["envs"] = validated_envs
                else:
                    self.logger.warning(f"Unexpected envs type: {type(envs)}, expected list")
            
            if command:
                payload["command"] = command
            
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully created instance: {data.get('id', 'unknown')}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, "Failed to create instance")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to create instance - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to create instance - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def start_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Start a stopped instance.
        
        Args:
            instance_id (str): Instance ID to start.
        
        Returns:
            dict: Response data with success status.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instance/start"
        try:
            payload = {"instanceId": instance_id}
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully started instance {instance_id}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, f"Failed to start instance {instance_id}")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to start instance - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to start instance - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Stop a running instance.
        
        Args:
            instance_id (str): Instance ID to stop.
        
        Returns:
            dict: Response data with success status.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instance/stop"
        try:
            payload = {"instanceId": instance_id}
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully stopped instance {instance_id}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, f"Failed to stop instance {instance_id}")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to stop instance - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to stop instance - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")
    
    def delete_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Delete an instance.
        
        Args:
            instance_id (str): Instance ID to delete.
        
        Returns:
            dict: Response data with success status.
            
        Raises:
            NovitaAuthenticationError: If API key is invalid.
            NovitaAPIError: If API request fails.
        """
        endpoint = f"{self.GPU_INSTANCE_BASE_URL}/gpu/instance/delete"
        try:
            payload = {"instanceId": instance_id}
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            if response.status_code == 401:
                self._log_response_error(response, "Authentication failed")
                raise NovitaAuthenticationError("Invalid API key")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully deleted instance {instance_id}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                self._log_response_error(e.response, f"Failed to delete instance {instance_id}")
                raise NovitaAPIError(
                    f"API request failed: HTTP {e.response.status_code} at {endpoint}"
                )
            else:
                self.logger.error(f"Failed to delete instance - No response available")
                raise NovitaAPIError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to delete instance - Request exception: {e}")
            raise NovitaAPIError(f"API request failed: {e}")

