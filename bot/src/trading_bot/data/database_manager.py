"""
Database Manager Module

Provides persistent SQLite storage for trades, positions, portfolio snapshots, and performance metrics.
Supports multi-timeframe analysis, transaction management, and automated backups.
"""

import sqlite3
import threading
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, event, exc as sqlalchemy_exc, text
from sqlalchemy.pool import QueuePool

from trading_bot.utils.helpers import retry_on_failure, ensure_dir
from trading_bot.utils.exceptions import DatabaseError
from trading_bot.utils.paths import get_writable_app_dir


class DatabaseManager:
    """
    Database manager for persistent storage of trading data.
    
    Handles trades, positions, portfolio snapshots, and performance metrics
    with thread-safe operations and connection pooling.
    
    Note: Connection Management
    This implementation uses SQLAlchemy's QueuePool for proper connection pooling.
    Connections are pooled and reused across threads, providing better resource
    management than thread-local connections. Transaction-scoped connections are
    maintained in thread-local storage to ensure atomicity.
    """
    
    def __init__(self, config, logger):
        """
        Initialize the database manager.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Get database configuration
        db_type = config.get('database.type', 'sqlite')
        db_path = config.get('database.path', 'data/trading_bot.db')
        
        # Determine database path
        if Path(db_path).is_absolute():
            self.db_path = db_path
        else:
            # Use writable app directory for data
            writable_dir = get_writable_app_dir('data')
            db_filename = Path(db_path).name
            self.db_path = os.path.join(writable_dir, db_filename)
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(self.db_path))
        
        # Connection pool settings
        pool_size = config.get('database.pool_size', config.get('database.connection_pool_size', 10))
        pool_max_overflow = config.get('database.pool_max_overflow', 20)
        pool_timeout = config.get('database.pool_timeout', 30)
        pool_recycle = config.get('database.pool_recycle', 3600)
        self.timeout = config.get('database.timeout_seconds', 30)
        self.enable_wal = config.get('database.enable_wal_mode', True)
        
        # Normalize Windows path to POSIX format for SQLite URL
        posix_path = Path(self.db_path).as_posix()
        
        # Construct SQLAlchemy database URL
        database_url = f"sqlite:///{posix_path}"
        
        # Create SQLAlchemy engine with QueuePool
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=pool_max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
            echo=False,
            connect_args={'check_same_thread': False, 'timeout': self.timeout}
        )
        
        # Register connect event to apply PRAGMAs on every new connection
        @event.listens_for(self.engine, 'connect')
        def on_connect(dbapi_conn, conn_record):
            """Apply connection-level PRAGMAs on every new DB-API connection."""
            # Set busy_timeout for this connection (in milliseconds)
            dbapi_conn.execute(f"PRAGMA busy_timeout={self.timeout * 1000}")
            # WAL mode is database-level, so it's set once in _initialize_database()
        
        # Threading lock for transactions
        self._lock = threading.Lock()
        
        # Thread-local storage for transaction-scoped connections
        # Used to ensure the same connection is used throughout a transaction
        self._transaction_local = threading.local()
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"Database manager initialized at: {self.db_path}")
    
    def _initialize_database(self):
        """Create database file and schema if it doesn't exist."""
        try:
            with self.get_connection() as conn:
                # Enable WAL mode for better concurrency (database-level, set once)
                if self.enable_wal:
                    conn.execute(text("PRAGMA journal_mode=WAL"))
                    conn.commit()
                    self.logger.debug("WAL mode enabled")
                
                # busy_timeout is now set via connect event for all connections
                
                # Create schema
                self._create_schema(conn)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(
                f"Database initialization failed: {str(e)}",
                details={'db_path': self.db_path, 'error': str(e)}
            )
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool.
        
        If we're in a transaction context, returns the transaction-scoped connection.
        Otherwise, gets a connection from the pool and returns it after use.
        
        Yields:
            SQLAlchemy Connection object (which provides SQLite-like API)
        """
        # Check if we're in a transaction context
        if hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None:
            yield self._transaction_local.connection
            return
        
        # Not in transaction - get connection from pool
        try:
            with self.engine.connect() as conn:
                # Set row factory for dict-like results
                # SQLAlchemy connections use text() for raw SQL, but we need to maintain compatibility
                # Wrap the connection to provide SQLite-like API
                yield conn
        except sqlalchemy_exc.TimeoutError as e:
            pool_timeout = self.config.get('database.pool_timeout', 30)
            raise DatabaseError(
                f"Connection pool exhausted (timeout after {pool_timeout}s). Consider increasing pool_size or max_overflow.",
                details={'error': str(e), 'pool_timeout': pool_timeout}
            )
        except sqlalchemy_exc.DBAPIError as e:
            if hasattr(e, 'connection_invalidated') and e.connection_invalidated:
                raise DatabaseError(
                    "Database connection lost. Reconnecting...",
                    details={'error': str(e)}
                )
            raise DatabaseError(
                f"Database operation failed: {str(e)}",
                details={'error': str(e)}
            )
        except Exception as e:
            raise DatabaseError(
                f"Database operation failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _create_schema(self, conn):
        """Create all database tables and indexes."""
        try:
            # Trades table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    timeframe TEXT,
                    strategy TEXT,
                    signal_type TEXT,
                    signal_confidence REAL,
                    notes TEXT
                )
            """))
            
            # Indexes for trades
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_timeframe ON trades(timeframe)"))
            # Composite index for symbol and exit_time ordering/filtering
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol_exit_time ON trades(symbol, exit_time)"))
            
            # Positions table
            # Note: For existing databases, the UNIQUE constraint on symbol may need to be removed manually
            # by recreating the table or using migration. For new databases, this schema uses composite uniqueness.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    timeframe TEXT,
                    strategy TEXT,
                    UNIQUE(symbol, timeframe, strategy)
                )
            """))
            
            # Indexes for positions
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON positions(entry_time)"))
            
            # Portfolio snapshots table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    num_positions INTEGER NOT NULL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    notes TEXT
                )
            """))
            
            # Index for snapshots
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp)"))
            
            # Performance metrics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    period TEXT NOT NULL CHECK(period IN ('daily', 'weekly', 'monthly', 'all_time')),
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win REAL,
                    avg_loss REAL,
                    total_return REAL
                )
            """))
            
            # Indexes for metrics
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metrics_period ON performance_metrics(period)"))
            
            # Feature importance table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    model_id TEXT,
                    symbol TEXT,
                    model_version TEXT,
                    training_date TEXT,
                    metadata TEXT
                )
            """))
            
            # Indexes for feature_importance
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feature_importance_timestamp ON feature_importance(timestamp)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feature_importance_feature_name ON feature_importance(feature_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feature_importance_symbol ON feature_importance(symbol)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feature_importance_model_id ON feature_importance(model_id)"))
            
            conn.commit()
            self.logger.debug("Database schema created successfully")
            
        except Exception as e:
            self.logger.error(f"Schema creation failed: {str(e)}")
            conn.rollback()
            raise DatabaseError(
                f"Schema creation failed: {str(e)}",
                details={'error': str(e)}
            )
    
    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(DatabaseError, sqlalchemy_exc.DBAPIError, sqlalchemy_exc.TimeoutError))
    def _execute_query(self, query: str, params: tuple = None, fetch: bool = False, commit: bool = True):
        """
        Execute a database query with retry logic.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            commit: Whether to commit the transaction (default: True). 
                    If False or if conn.in_transaction is True, commit is skipped.
            
        Returns:
            Query results if fetch=True, else None
        """
        try:
            with self.get_connection() as conn:
                # Use exec_driver_sql for parameterized queries to properly handle ? placeholders
                if params:
                    result = conn.exec_driver_sql(query, params)
                else:
                    result = conn.exec_driver_sql(query)
                
                if fetch:
                    # Convert SQLAlchemy Row objects to dict-like objects
                    # SQLAlchemy 2.0 returns Row objects that are dict-like
                    rows = result.fetchall()
                    # Convert to list of dicts for backward compatibility
                    return [dict(row._mapping) for row in rows]
                else:
                    # Only commit if explicitly requested and not in a transaction
                    # Check if we're in a transaction context (stored in thread-local)
                    in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
                    if commit and not in_transaction:
                        conn.commit()
                    # Get lastrowid from result
                    return result.lastrowid if hasattr(result, 'lastrowid') and result.lastrowid else None
                    
        except sqlalchemy_exc.TimeoutError as e:
            pool_timeout = self.config.get('database.pool_timeout', 30)
            self.logger.error(f"Query execution failed: Pool exhausted (timeout after {pool_timeout}s)")
            raise DatabaseError(
                f"Connection pool exhausted (timeout after {pool_timeout}s). Consider increasing pool_size or max_overflow.",
                details={'query': query[:100], 'error': str(e), 'pool_timeout': pool_timeout}
            )
        except sqlalchemy_exc.DBAPIError as e:
            if hasattr(e, 'connection_invalidated') and e.connection_invalidated:
                self.logger.error(f"Query execution failed: Connection invalidated")
                raise DatabaseError(
                    "Database connection lost. Reconnecting...",
                    details={'query': query[:100], 'error': str(e)}
                )
            self.logger.error(f"Query execution failed: {str(e)}, Query: {query[:100]}")
            raise DatabaseError(
                f"Query execution failed: {str(e)}",
                details={'query': query[:100], 'error': str(e)}
            )
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}, Query: {query[:100]}")
            raise DatabaseError(
                f"Query execution failed: {str(e)}",
                details={'query': query[:100], 'error': str(e)}
            )
    
    # Transaction Management
    
    def begin_transaction(self):
        """Begin a database transaction."""
        # Acquire a connection from the pool
        conn = self.engine.connect()
        # Store it in thread-local for transaction scope
        self._transaction_local.connection = conn
        # Begin transaction
        conn.begin()
    
    def commit_transaction(self):
        """Commit the current transaction."""
        # Get connection from thread-local
        conn = self._transaction_local.connection
        if conn is None:
            raise DatabaseError("No active transaction to commit")
        try:
            conn.commit()
        finally:
            # Close connection (returns to pool)
            conn.close()
            # Clear thread-local
            self._transaction_local.connection = None
    
    def rollback_transaction(self):
        """Rollback the current transaction."""
        # Get connection from thread-local
        conn = self._transaction_local.connection
        if conn is None:
            raise DatabaseError("No active transaction to rollback")
        try:
            conn.rollback()
        finally:
            # Close connection (returns to pool)
            conn.close()
            # Clear thread-local
            self._transaction_local.connection = None
    
    @contextmanager
    def transaction(self):
        """Context manager for automatic transaction handling."""
        try:
            self.begin_transaction()
            yield
            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise
    
    # Trades CRUD Operations
    
    def insert_trade(self, symbol: str, side: str, entry_price: float, 
                    exit_price: float, quantity: float, entry_time: str,
                    exit_time: str, pnl: float = None, pnl_percentage: float = None,
                    timeframe: str = None, strategy: str = None,
                    signal_type: str = None, signal_confidence: float = None,
                    notes: str = None, commit: bool = True) -> int:
        """
        Insert a completed trade record.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            entry_time: Entry timestamp (ISO8601)
            exit_time: Exit timestamp (ISO8601)
            pnl: Profit/loss (calculated if not provided)
            pnl_percentage: PnL percentage (calculated if not provided)
            timeframe: Trading timeframe
            strategy: Strategy name
            signal_type: Signal type
            signal_confidence: Signal confidence
            notes: Optional notes
            
        Returns:
            Trade ID
        """
        # Validate timeframe if provided
        if timeframe is not None and not self._validate_timeframe(timeframe):
            raise DatabaseError(
                f"Invalid timeframe: {timeframe}",
                details={'timeframe': timeframe, 'valid_timeframes': self.config.get('data.timeframes', [])}
            )
        
        # Calculate PnL if not provided
        if pnl is None:
            multiplier = 1 if side == 'BUY' else -1
            pnl = (exit_price - entry_price) * quantity * multiplier
        
        if pnl_percentage is None:
            pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
            if side == 'SELL':
                pnl_percentage = -pnl_percentage
        
        query = """
            INSERT INTO trades (
                symbol, side, entry_price, exit_price, quantity,
                entry_time, exit_time, pnl, pnl_percentage,
                timeframe, strategy, signal_type, signal_confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            symbol, side, entry_price, exit_price, quantity,
            entry_time, exit_time, pnl, pnl_percentage,
            timeframe, strategy, signal_type, signal_confidence, notes
        )
        
        trade_id = self._execute_query(query, params, commit=commit)
        self.logger.info(f"Trade inserted: {symbol} {side} (ID: {trade_id})")
        return trade_id
    
    def get_trade_by_id(self, trade_id: int) -> Optional[Dict]:
        """Fetch trade by ID."""
        query = "SELECT * FROM trades WHERE id = ?"
        results = self._execute_query(query, (trade_id,), fetch=True)
        
        if results:
            return dict(results[0])
        return None
    
    def get_trades(self, symbol: str = None, start_date: str = None,
                   end_date: str = None, timeframe: str = None,
                   limit: int = None) -> List[Dict]:
        """Query trades with filters."""
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"SELECT * FROM trades {where_clause} ORDER BY exit_time DESC {limit_clause}"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [dict(row) for row in results]
    
    def get_all_trades(self, limit: int = None, offset: int = None) -> List[Dict]:
        """Get paginated trade history."""
        limit_clause = f"LIMIT {limit}" if limit else ""
        offset_clause = f"OFFSET {offset}" if offset else ""
        
        query = f"SELECT * FROM trades ORDER BY exit_time DESC {limit_clause} {offset_clause}"
        results = self._execute_query(query, fetch=True)
        return [dict(row) for row in results]
    
    def get_trades_by_symbol(self, symbol: str, limit: int = None) -> List[Dict]:
        """Get all trades for a symbol."""
        return self.get_trades(symbol=symbol, limit=limit)
    
    def get_trades_by_timeframe(self, timeframe: str, limit: int = None) -> List[Dict]:
        """Get all trades for a timeframe."""
        return self.get_trades(timeframe=timeframe, limit=limit)
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get most recent trades."""
        return self.get_all_trades(limit=limit)
    
    def update_trade(self, trade_id: int, **kwargs) -> bool:
        """Update specific fields of a trade."""
        allowed_fields = [
            'symbol', 'side', 'entry_price', 'exit_price', 'quantity',
            'entry_time', 'exit_time', 'pnl', 'pnl_percentage',
            'timeframe', 'strategy', 'signal_type', 'signal_confidence', 'notes'
        ]
        
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        params.append(trade_id)
        query = f"UPDATE trades SET {', '.join(updates)} WHERE id = ?"
        
        self._execute_query(query, tuple(params))
        self.logger.debug(f"Trade {trade_id} updated")
        return True
    
    def delete_trade(self, trade_id: int) -> int:
        """Delete a trade record."""
        query = "DELETE FROM trades WHERE id = ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (trade_id,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        self.logger.info(f"Trade {trade_id} deleted")
        return deleted_count
    
    def delete_trades_before(self, date: str) -> int:
        """Bulk delete old trades."""
        query = "DELETE FROM trades WHERE exit_time < ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (date,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        return deleted_count
    
    # Trade Aggregation Queries
    
    def get_trade_statistics(self, symbol: str = None, start_date: str = None,
                            end_date: str = None) -> Dict:
        """Get trade statistics."""
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_win,
                MIN(pnl) as max_loss
            FROM trades
            {where_clause}
        """
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        
        if results and results[0]:
            row = dict(results[0])
            total = row.get('total_trades', 0) or 0
            wins = row.get('winning_trades', 0) or 0
            
            return {
                'total_trades': total,
                'winning_trades': wins,
                'losing_trades': row.get('losing_trades', 0) or 0,
                'total_pnl': row.get('total_pnl', 0.0) or 0.0,
                'win_rate': (wins / total * 100) if total > 0 else 0.0,
                'avg_pnl': row.get('avg_pnl', 0.0) or 0.0,
                'max_win': row.get('max_win', 0.0) or 0.0,
                'max_loss': row.get('max_loss', 0.0) or 0.0
            }
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0
        }
    
    def get_pnl_by_symbol(self) -> Dict[str, float]:
        """Get total PnL grouped by symbol."""
        query = "SELECT symbol, SUM(pnl) as total_pnl FROM trades GROUP BY symbol"
        results = self._execute_query(query, fetch=True)
        return {row['symbol']: row['total_pnl'] for row in results}
    
    def get_pnl_by_timeframe(self) -> Dict[str, float]:
        """Get total PnL grouped by timeframe."""
        query = "SELECT timeframe, SUM(pnl) as total_pnl FROM trades WHERE timeframe IS NOT NULL GROUP BY timeframe"
        results = self._execute_query(query, fetch=True)
        return {row['timeframe']: row['total_pnl'] for row in results}
    
    # Positions CRUD Operations
    
    def insert_position(self, symbol: str, side: str, entry_price: float,
                       quantity: float, stop_loss: float = None,
                       take_profit: float = None, entry_time: str = None,
                       timeframe: str = None, strategy: str = None) -> int:
        """Open a new position."""
        if entry_time is None:
            entry_time = datetime.now().isoformat()
        
        # Validate timeframe if provided
        if timeframe is not None and not self._validate_timeframe(timeframe):
            raise DatabaseError(
                f"Invalid timeframe: {timeframe}",
                details={'timeframe': timeframe, 'valid_timeframes': self.config.get('data.timeframes', [])}
            )
        
        # Check for existing position with same (symbol, timeframe, strategy)
        existing = self._check_position_exists(symbol, timeframe, strategy)
        if existing:
            raise DatabaseError(
                f"Position already exists for symbol {symbol}, timeframe {timeframe}, strategy {strategy}",
                details={'symbol': symbol, 'timeframe': timeframe, 'strategy': strategy}
            )
        
        current_price = entry_price
        unrealized_pnl = 0.0
        last_updated = entry_time
        
        query = """
            INSERT INTO positions (
                symbol, side, entry_price, quantity, current_price,
                unrealized_pnl, stop_loss, take_profit, entry_time,
                last_updated, timeframe, strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            symbol, side, entry_price, quantity, current_price,
            unrealized_pnl, stop_loss, take_profit, entry_time,
            last_updated, timeframe, strategy
        )
        
        position_id = self._execute_query(query, params)
        self.logger.info(f"Position opened: {symbol} {side} (ID: {position_id})")
        return position_id
    
    def _check_position_exists(self, symbol: str, timeframe: str = None, strategy: str = None) -> bool:
        """Check if a position exists with the given composite key."""
        conditions = ["symbol = ?"]
        params = [symbol]
        
        if timeframe is not None:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        else:
            conditions.append("timeframe IS NULL")
        
        if strategy is not None:
            conditions.append("strategy = ?")
            params.append(strategy)
        else:
            conditions.append("strategy IS NULL")
        
        where_clause = "WHERE " + " AND ".join(conditions)
        query = f"SELECT COUNT(*) as count FROM positions {where_clause}"
        
        results = self._execute_query(query, tuple(params), fetch=True)
        return results and results[0]['count'] > 0 if results else False
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Fetch position by symbol."""
        query = "SELECT * FROM positions WHERE symbol = ?"
        results = self._execute_query(query, (symbol,), fetch=True)
        
        if results:
            return dict(results[0])
        return None
    
    def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        query = "SELECT * FROM positions ORDER BY entry_time DESC"
        results = self._execute_query(query, fetch=True)
        return [dict(row) for row in results]
    
    def get_positions_by_timeframe(self, timeframe: str) -> List[Dict]:
        """Get positions filtered by timeframe."""
        query = "SELECT * FROM positions WHERE timeframe = ? ORDER BY entry_time DESC"
        results = self._execute_query(query, (timeframe,), fetch=True)
        return [dict(row) for row in results]
    
    def get_position_count(self) -> int:
        """Get count of open positions."""
        query = "SELECT COUNT(*) as count FROM positions"
        results = self._execute_query(query, fetch=True)
        return results[0]['count'] if results else 0
    
    def update_position_price(self, symbol: str, current_price: float) -> bool:
        """Update position price and recalculate unrealized PnL."""
        # Get position
        position = self.get_position_by_symbol(symbol)
        if not position:
            return False
        
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        # Calculate unrealized PnL
        multiplier = 1 if side == 'BUY' else -1
        unrealized_pnl = (current_price - entry_price) * quantity * multiplier
        
        query = """
            UPDATE positions 
            SET current_price = ?, unrealized_pnl = ?, last_updated = ?
            WHERE symbol = ?
        """
        
        params = (current_price, unrealized_pnl, datetime.now().isoformat(), symbol)
        self._execute_query(query, params)
        return True
    
    def update_position_stops(self, symbol: str, stop_loss: float = None,
                             take_profit: float = None) -> bool:
        """Update stop-loss/take-profit levels."""
        updates = []
        params = []
        
        if stop_loss is not None:
            updates.append("stop_loss = ?")
            params.append(stop_loss)
        
        if take_profit is not None:
            updates.append("take_profit = ?")
            params.append(take_profit)
        
        if not updates:
            return False
        
        params.append(symbol)
        query = f"UPDATE positions SET {', '.join(updates)} WHERE symbol = ?"
        
        self._execute_query(query, tuple(params))
        return True
    
    def update_position(self, symbol: str, **kwargs) -> bool:
        """Generic position update."""
        allowed_fields = [
            'side', 'entry_price', 'quantity', 'current_price',
            'unrealized_pnl', 'stop_loss', 'take_profit',
            'entry_time', 'last_updated', 'timeframe', 'strategy'
        ]
        
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        params.append(symbol)
        query = f"UPDATE positions SET {', '.join(updates)} WHERE symbol = ?"
        
        self._execute_query(query, tuple(params))
        return True
    
    def close_position(self, symbol: str, exit_price: float,
                      exit_time: str = None, commission: float = None) -> Optional[int]:
        """
        Close position and move to trades table.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_time: Exit timestamp (ISO8601)
            commission: Commission paid (will be subtracted from PnL and stored in notes)
        
        Returns:
            Trade ID if successful, None otherwise
        """
        if exit_time is None:
            exit_time = datetime.now().isoformat()
        
        try:
            with self.transaction():
                # Get position
                position = self.get_position_by_symbol(symbol)
                if not position:
                    return None
                
                # Calculate final PnL
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side']
                
                multiplier = 1 if side == 'BUY' else -1
                pnl = (exit_price - entry_price) * quantity * multiplier
                
                # Subtract commission from PnL if provided
                if commission is not None:
                    pnl -= commission
                
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
                if side == 'SELL':
                    pnl_percentage = -pnl_percentage
                
                # Build notes with commission info
                notes = f"Closed position (ID: {position['id']})"
                if commission is not None:
                    notes += f"; Commission: ${commission:.2f}"
                
                # Insert into trades
                trade_id = self.insert_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_time=position['entry_time'],
                    exit_time=exit_time,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    timeframe=position.get('timeframe'),
                    strategy=position.get('strategy'),
                    notes=notes,
                    commit=False
                )
                
                # Delete from positions
                self.delete_position(symbol, commit=False)
                
                self.logger.info(f"Position closed: {symbol} -> Trade ID: {trade_id}")
                return trade_id
                
        except Exception as e:
            self.logger.error(f"Failed to close position {symbol}: {str(e)}")
            raise DatabaseError(
                f"Failed to close position: {str(e)}",
                details={'symbol': symbol, 'error': str(e)}
            )
    
    def delete_position(self, symbol: str, commit: bool = True) -> int:
        """Force delete position without creating trade record."""
        query = "DELETE FROM positions WHERE symbol = ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (symbol,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if commit and not in_transaction:
                conn.commit()
        self.logger.info(f"Position deleted: {symbol}")
        return deleted_count
    
    def update_all_positions_prices(self, price_dict: Dict[str, float]) -> int:
        """Update multiple positions at once."""
        updated = 0
        for symbol, price in price_dict.items():
            if self.update_position_price(symbol, price):
                updated += 1
        return updated
    
    def get_total_positions_value(self) -> float:
        """Get total value of all positions."""
        query = "SELECT SUM(quantity * current_price) as total_value FROM positions"
        results = self._execute_query(query, fetch=True)
        return results[0]['total_value'] if results and results[0]['total_value'] else 0.0
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        query = "SELECT SUM(unrealized_pnl) as total_pnl FROM positions"
        results = self._execute_query(query, fetch=True)
        return results[0]['total_pnl'] if results and results[0]['total_pnl'] else 0.0
    
    # Portfolio Snapshots CRUD Operations
    
    def insert_portfolio_snapshot(self, total_equity: float, cash_balance: float,
                                 positions_value: float, num_positions: int,
                                 daily_pnl: float = None, total_pnl: float = None,
                                 notes: str = None) -> int:
        """Record portfolio state."""
        timestamp = datetime.now().isoformat()
        
        query = """
            INSERT INTO portfolio_snapshots (
                timestamp, total_equity, cash_balance, positions_value,
                num_positions, daily_pnl, total_pnl, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            timestamp, total_equity, cash_balance, positions_value,
            num_positions, daily_pnl, total_pnl, notes
        )
        
        snapshot_id = self._execute_query(query, params)
        self.logger.debug(f"Portfolio snapshot created (ID: {snapshot_id})")
        return snapshot_id
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get most recent portfolio snapshot."""
        query = "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        results = self._execute_query(query, fetch=True)
        
        if results:
            return dict(results[0])
        return None
    
    def get_snapshots(self, start_date: str = None, end_date: str = None,
                     limit: int = None) -> List[Dict]:
        """Get time-filtered snapshots."""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"SELECT * FROM portfolio_snapshots {where_clause} ORDER BY timestamp DESC {limit_clause}"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [dict(row) for row in results]
    
    def get_snapshots_by_period(self, period: str = 'daily', limit: int = 30) -> List[Dict]:
        """Get snapshots for specific period."""
        # Simple implementation - can be enhanced with date grouping
        return self.get_snapshots(limit=limit)
    
    def get_snapshot_by_id(self, snapshot_id: int) -> Optional[Dict]:
        """Fetch specific snapshot."""
        query = "SELECT * FROM portfolio_snapshots WHERE id = ?"
        results = self._execute_query(query, (snapshot_id,), fetch=True)
        
        if results:
            return dict(results[0])
        return None
    
    def update_snapshot(self, snapshot_id: int, **kwargs) -> bool:
        """Update snapshot fields."""
        allowed_fields = [
            'timestamp', 'total_equity', 'cash_balance', 'positions_value',
            'num_positions', 'daily_pnl', 'total_pnl', 'notes'
        ]
        
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        params.append(snapshot_id)
        query = f"UPDATE portfolio_snapshots SET {', '.join(updates)} WHERE id = ?"
        
        self._execute_query(query, tuple(params))
        return True
    
    def delete_snapshots_before(self, date: str) -> int:
        """Delete old snapshots."""
        query = "DELETE FROM portfolio_snapshots WHERE timestamp < ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (date,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        return deleted_count
    
    def delete_snapshot(self, snapshot_id: int) -> int:
        """Delete specific snapshot."""
        query = "DELETE FROM portfolio_snapshots WHERE id = ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (snapshot_id,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        return deleted_count
    
    def get_equity_curve(self, start_date: str = None, end_date: str = None) -> List[Tuple[str, float]]:
        """Get equity curve data for plotting."""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"SELECT timestamp, total_equity FROM portfolio_snapshots {where_clause} ORDER BY timestamp ASC"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [(row['timestamp'], row['total_equity']) for row in results]
    
    def get_portfolio_growth(self, start_date: str, end_date: str) -> float:
        """Calculate portfolio growth percentage."""
        start_snap = self.get_snapshots(end_date=start_date, limit=1)
        end_snap = self.get_snapshots(end_date=end_date, limit=1)
        
        if not start_snap or not end_snap:
            return 0.0
        
        start_equity = start_snap[0]['total_equity']
        end_equity = end_snap[0]['total_equity']
        
        if start_equity == 0:
            return 0.0
        
        return ((end_equity - start_equity) / start_equity) * 100
    
    def get_drawdown_periods(self) -> List[Dict]:
        """Identify drawdown periods."""
        snapshots = self.get_snapshots()
        if not snapshots:
            return []
        
        # Find peak equity and calculate drawdowns
        peaks = []
        max_equity = 0.0
        
        for snap in snapshots:
            equity = snap['total_equity']
            if equity > max_equity:
                max_equity = equity
            
            drawdown = ((equity - max_equity) / max_equity) * 100 if max_equity > 0 else 0.0
            
            peaks.append({
                'timestamp': snap['timestamp'],
                'equity': equity,
                'peak_equity': max_equity,
                'drawdown': drawdown
            })
        
        return peaks
    
    # Performance Metrics CRUD Operations
    
    def insert_performance_metrics(self, period: str, sharpe_ratio: float = None,
                                  sortino_ratio: float = None, max_drawdown: float = None,
                                  win_rate: float = None, profit_factor: float = None,
                                  total_trades: int = None, winning_trades: int = None,
                                  losing_trades: int = None, avg_win: float = None,
                                  avg_loss: float = None, total_return: float = None) -> int:
        """Store calculated performance metrics."""
        timestamp = datetime.now().isoformat()
        
        query = """
            INSERT INTO performance_metrics (
                timestamp, period, sharpe_ratio, sortino_ratio, max_drawdown,
                win_rate, profit_factor, total_trades, winning_trades, losing_trades,
                avg_win, avg_loss, total_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            timestamp, period, sharpe_ratio, sortino_ratio, max_drawdown,
            win_rate, profit_factor, total_trades, winning_trades, losing_trades,
            avg_win, avg_loss, total_return
        )
        
        metrics_id = self._execute_query(query, params)
        self.logger.debug(f"Performance metrics stored (ID: {metrics_id}, period: {period})")
        return metrics_id
    
    def get_latest_metrics(self, period: str = 'daily') -> Optional[Dict]:
        """Get most recent metrics for period."""
        query = "SELECT * FROM performance_metrics WHERE period = ? ORDER BY timestamp DESC LIMIT 1"
        results = self._execute_query(query, (period,), fetch=True)
        
        if results:
            return dict(results[0])
        return None
    
    def get_metrics_history(self, period: str = 'daily', limit: int = 30) -> List[Dict]:
        """Get historical metrics."""
        query = "SELECT * FROM performance_metrics WHERE period = ? ORDER BY timestamp DESC LIMIT ?"
        results = self._execute_query(query, (period, limit), fetch=True)
        return [dict(row) for row in results]
    
    def get_metrics_by_date_range(self, start_date: str, end_date: str,
                                  period: str = 'daily') -> List[Dict]:
        """Get time-filtered metrics."""
        query = """
            SELECT * FROM performance_metrics 
            WHERE period = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """
        results = self._execute_query(query, (period, start_date, end_date), fetch=True)
        return [dict(row) for row in results]
    
    def get_all_metrics(self, limit: int = None) -> List[Dict]:
        """Get all stored metrics."""
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"SELECT * FROM performance_metrics ORDER BY timestamp DESC {limit_clause}"
        results = self._execute_query(query, fetch=True)
        return [dict(row) for row in results]
    
    def update_metrics(self, metrics_id: int, **kwargs) -> bool:
        """Update metric values."""
        allowed_fields = [
            'timestamp', 'period', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'win_rate', 'profit_factor', 'total_trades', 'winning_trades',
            'losing_trades', 'avg_win', 'avg_loss', 'total_return'
        ]
        
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        params.append(metrics_id)
        query = f"UPDATE performance_metrics SET {', '.join(updates)} WHERE id = ?"
        
        self._execute_query(query, tuple(params))
        return True
    
    def delete_metrics_before(self, date: str) -> int:
        """Delete old metrics."""
        query = "DELETE FROM performance_metrics WHERE timestamp < ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (date,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        return deleted_count
    
    def delete_metrics(self, metrics_id: int) -> int:
        """Delete specific metrics."""
        query = "DELETE FROM performance_metrics WHERE id = ?"
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (metrics_id,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        return deleted_count
    
    def compare_metrics_periods(self, period1: str, period2: str) -> Dict:
        """Compare metrics between two periods."""
        metrics1 = self.get_latest_metrics(period1)
        metrics2 = self.get_latest_metrics(period2)
        
        if not metrics1 or not metrics2:
            return {}
        
        comparison = {}
        for key in ['sharpe_ratio', 'sortino_ratio', 'win_rate', 'total_return']:
            val1 = metrics1.get(key) or 0.0
            val2 = metrics2.get(key) or 0.0
            comparison[key] = {
                period1: val1,
                period2: val2,
                'difference': val2 - val1
            }
        
        return comparison
    
    # Feature Importance Methods
    
    def insert_feature_importance(self, timestamp: str, feature_name: str, 
                                 importance_score: float, model_id: Optional[str] = None,
                                 symbol: Optional[str] = None, model_version: Optional[str] = None,
                                 training_date: Optional[str] = None, metadata: Optional[str] = None) -> int:
        """
        Insert a single feature importance record.
        
        Args:
            timestamp: Timestamp for tracking
            feature_name: Name of the feature
            importance_score: Importance score
            model_id: Optional model identifier
            symbol: Optional symbol
            model_version: Optional model version
            training_date: Optional training date
            metadata: Optional JSON metadata string
        
        Returns:
            ID of inserted record
        """
        query = """
            INSERT INTO feature_importance 
            (timestamp, feature_name, importance_score, model_id, symbol, model_version, training_date, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.get_connection() as conn:
                result = conn.exec_driver_sql(
                    query,
                    (timestamp, feature_name, importance_score, model_id, symbol, 
                     model_version, training_date, metadata)
                )
                in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
                if not in_transaction:
                    conn.commit()
                return result.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to insert feature importance: {str(e)}")
            raise DatabaseError(f"Failed to insert feature importance: {str(e)}")
    
    def insert_feature_importance_batch(self, records: List[Dict]) -> int:
        """
        Batch insert feature importance records.
        
        Args:
            records: List of dictionaries with feature importance data
        
        Returns:
            Number of records inserted
        """
        if not records:
            return 0
        
        query = """
            INSERT INTO feature_importance 
            (timestamp, feature_name, importance_score, model_id, symbol, model_version, training_date, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            # Prepare data tuples
            data_tuples = []
            for record in records:
                data_tuples.append((
                    record.get('timestamp'),
                    record.get('feature_name'),
                    record.get('importance_score'),
                    record.get('model_id'),
                    record.get('symbol'),
                    record.get('model_version'),
                    record.get('training_date'),
                    record.get('metadata')
                ))
            
            with self.get_connection() as conn:
                result = conn.exec_driver_sql(query, data_tuples)
                in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
                if not in_transaction:
                    conn.commit()
                
                inserted_count = result.rowcount if hasattr(result, 'rowcount') else len(data_tuples)
                self.logger.debug(f"Inserted {inserted_count} feature importance records")
                return inserted_count
                
        except Exception as e:
            self.logger.error(f"Failed to batch insert feature importance: {str(e)}")
            raise DatabaseError(f"Failed to batch insert feature importance: {str(e)}")
    
    def query_feature_importance(self, feature_name: Optional[str] = None,
                                symbol: Optional[str] = None, model_id: Optional[str] = None,
                                start_date: Optional[str] = None, end_date: Optional[str] = None,
                                limit: Optional[int] = None) -> List[Dict]:
        """
        Query feature importance with optional filters.
        
        Args:
            feature_name: Optional feature name filter
            symbol: Optional symbol filter
            model_id: Optional model ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit for results
        
        Returns:
            List of dictionaries with feature importance records
        """
        query = "SELECT * FROM feature_importance WHERE 1=1"
        params = []
        
        if feature_name:
            query += " AND feature_name = ?"
            params.append(feature_name)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            results = self._execute_query(query, tuple(params) if params else None, fetch=True)
            return [dict(row) for row in results] if results else []
        except Exception as e:
            self.logger.error(f"Failed to query feature importance: {str(e)}")
            return []
    
    def get_latest_feature_importance(self, symbol: Optional[str] = None,
                                     model_id: Optional[str] = None, n: int = 100) -> List[Dict]:
        """
        Get most recent feature importance records.
        
        Args:
            symbol: Optional symbol filter
            model_id: Optional model ID filter
            n: Number of records to return
        
        Returns:
            List of dictionaries with recent feature importance records
        """
        return self.query_feature_importance(symbol=symbol, model_id=model_id, limit=n)
    
    def get_feature_importance_history(self, feature_name: str, symbol: Optional[str] = None,
                                      days: int = 90) -> List[Dict]:
        """
        Get historical importance scores for a specific feature.
        
        Args:
            feature_name: Name of the feature
            symbol: Optional symbol filter
            days: Number of days to look back
        
        Returns:
            List of dictionaries with historical importance records
        """
        end_date = datetime.now().isoformat()
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        return self.query_feature_importance(
            feature_name=feature_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
    
    def delete_old_feature_importance(self, days: int = 90) -> int:
        """
        Delete feature importance records older than N days.
        
        Args:
            days: Number of days to keep
        
        Returns:
            Number of records deleted
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        query = "DELETE FROM feature_importance WHERE timestamp < ?"
        
        try:
            with self.get_connection() as conn:
                result = conn.exec_driver_sql(query, (cutoff_date,))
                deleted_count = result.rowcount
                in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
                if not in_transaction:
                    conn.commit()
                
                self.logger.info(f"Deleted {deleted_count} old feature importance records")
                return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to delete old feature importance: {str(e)}")
            return 0
    
    def get_metrics_trend(self, metric_name: str, period: str = 'daily',
                         limit: int = 30) -> List[Dict]:
        """Get trend for specific metric."""
        metrics = self.get_metrics_history(period, limit)
        trend = []
        
        for metric in metrics:
            if metric_name in metric:
                trend.append({
                    'timestamp': metric['timestamp'],
                    'value': metric[metric_name]
                })
        
        return trend
    
    # Time-based Filtering Utilities
    
    def _parse_date_filter(self, date_str: str) -> str:
        """Parse date string to ISO8601 format."""
        # Assume date_str is already in ISO8601 or can be parsed
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.isoformat()
        except:
            return date_str
    
    def _build_date_filter_clause(self, start_date: str = None,
                                  end_date: str = None,
                                  column_name: str = 'timestamp') -> Tuple[str, List]:
        """Build SQL WHERE clause for date ranges."""
        conditions = []
        params = []
        
        if start_date:
            conditions.append(f"{column_name} >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append(f"{column_name} <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params
    
    def _validate_timeframe(self, timeframe: str) -> bool:
        """Validate timeframe against config."""
        valid_timeframes = self.config.get('data.timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        return timeframe in valid_timeframes
    
    def get_trades_grouped_by_timeframe(self, start_date: str = None,
                                       end_date: str = None) -> Dict[str, List[Dict]]:
        """Get trades grouped by timeframe."""
        trades = self.get_trades(start_date=start_date, end_date=end_date)
        
        grouped = {}
        for trade in trades:
            tf = trade.get('timeframe', 'unknown')
            if tf not in grouped:
                grouped[tf] = []
            grouped[tf].append(trade)
        
        return grouped
    
    def get_performance_by_timeframe(self, timeframe: str, start_date: str = None,
                                    end_date: str = None) -> Dict:
        """Calculate performance for specific timeframe."""
        # Compute stats scoped to the specific timeframe
        conditions = ["timeframe = ?"]
        params = [timeframe]
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_win,
                MIN(pnl) as max_loss
            FROM trades
            {where_clause}
        """
        
        results = self._execute_query(query, tuple(params), fetch=True)
        
        if results and results[0]:
            row = dict(results[0])
            total = row.get('total_trades', 0) or 0
            wins = row.get('winning_trades', 0) or 0
            
            return {
                'timeframe': timeframe,
                'total_trades': total,
                'win_rate': (wins / total * 100) if total > 0 else 0.0,
                'total_pnl': row.get('total_pnl', 0.0) or 0.0,
                'avg_pnl': row.get('avg_pnl', 0.0) or 0.0
            }
        
        return {
            'timeframe': timeframe,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0
        }
    
    def get_best_performing_timeframe(self, start_date: str = None,
                                     end_date: str = None) -> Optional[str]:
        """Get timeframe with highest win rate."""
        timeframes = self.config.get('data.timeframes', [])
        best_tf = None
        best_win_rate = 0.0
        
        for tf in timeframes:
            perf = self.get_performance_by_timeframe(tf, start_date, end_date)
            if perf['win_rate'] > best_win_rate:
                best_win_rate = perf['win_rate']
                best_tf = tf
        
        return best_tf
    
    def get_daily_pnl(self, start_date: str = None, end_date: str = None) -> List[Tuple[str, float]]:
        """Get daily PnL aggregation."""
        query = """
            SELECT DATE(exit_time) as date, SUM(pnl) as daily_pnl
            FROM trades
        """
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY DATE(exit_time) ORDER BY date ASC"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [(row['date'], row['daily_pnl'] or 0.0) for row in results]
    
    def get_weekly_pnl(self, start_date: str = None, end_date: str = None) -> List[Tuple[str, float]]:
        """Get weekly PnL aggregation."""
        # SQLite doesn't have WEEK() function, use strftime
        query = """
            SELECT strftime('%Y-W%W', exit_time) as week, SUM(pnl) as weekly_pnl
            FROM trades
        """
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY strftime('%Y-W%W', exit_time) ORDER BY week ASC"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [(row['week'], row['weekly_pnl'] or 0.0) for row in results]
    
    def get_monthly_pnl(self, start_date: str = None, end_date: str = None) -> List[Tuple[str, float]]:
        """Get monthly PnL aggregation."""
        query = """
            SELECT strftime('%Y-%m', exit_time) as month, SUM(pnl) as monthly_pnl
            FROM trades
        """
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY strftime('%Y-%m', exit_time) ORDER BY month ASC"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [(row['month'], row['monthly_pnl'] or 0.0) for row in results]
    
    # Feature Importance CRUD Operations
    
    def insert_feature_importance(self, timestamp: str, feature_name: str, 
                                 importance_score: float, model_id: Optional[str] = None,
                                 symbol: Optional[str] = None, model_version: Optional[str] = None,
                                 training_date: Optional[str] = None, metadata: Optional[str] = None) -> int:
        """
        Insert a single feature importance record.
        
        Args:
            timestamp: Timestamp for tracking
            feature_name: Name of the feature
            importance_score: Importance score (0-1)
            model_id: Optional model identifier
            symbol: Optional trading symbol
            model_version: Optional model version
            training_date: Optional training date
            metadata: Optional JSON metadata string
        
        Returns:
            Record ID
        """
        query = """
            INSERT INTO feature_importance (
                timestamp, feature_name, importance_score, model_id,
                symbol, model_version, training_date, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (timestamp, feature_name, importance_score, model_id,
                 symbol, model_version, training_date, metadata)
        
        record_id = self._execute_query(query, params)
        self.logger.debug(f"Feature importance inserted: {feature_name} (ID: {record_id})")
        return record_id
    
    def insert_feature_importance_batch(self, records: List[Dict]) -> int:
        """
        Batch insert feature importance records.
        
        Args:
            records: List of dictionaries with feature importance data
        
        Returns:
            Number of records inserted
        """
        if not records:
            return 0
        
        query = """
            INSERT INTO feature_importance (
                timestamp, feature_name, importance_score, model_id,
                symbol, model_version, training_date, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.transaction():
                inserted = 0
                for record in records:
                    params = (
                        record.get('timestamp'),
                        record.get('feature_name'),
                        record.get('importance_score'),
                        record.get('model_id'),
                        record.get('symbol'),
                        record.get('model_version'),
                        record.get('training_date'),
                        record.get('metadata')
                    )
                    self._execute_query(query, params, commit=False)
                    inserted += 1
                
                self.logger.info(f"Batch inserted {inserted} feature importance records")
                return inserted
        except Exception as e:
            self.logger.error(f"Batch insert failed: {str(e)}")
            raise
    
    def query_feature_importance(self, feature_name: Optional[str] = None,
                                symbol: Optional[str] = None, model_id: Optional[str] = None,
                                start_date: Optional[str] = None, end_date: Optional[str] = None,
                                limit: Optional[int] = None) -> List[Dict]:
        """
        Query feature importance records with filters.
        
        Args:
            feature_name: Filter by feature name
            symbol: Filter by symbol
            model_id: Filter by model ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records to return
        
        Returns:
            List of feature importance records
        """
        conditions = []
        params = []
        
        if feature_name:
            conditions.append("feature_name = ?")
            params.append(feature_name)
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"SELECT * FROM feature_importance {where_clause} ORDER BY timestamp DESC {limit_clause}"
        
        results = self._execute_query(query, tuple(params) if params else None, fetch=True)
        return [dict(row) for row in results]
    
    def get_latest_feature_importance(self, symbol: Optional[str] = None,
                                     model_id: Optional[str] = None, n: int = 1) -> List[Dict]:
        """
        Get latest feature importance records.
        
        Args:
            symbol: Optional symbol filter
            model_id: Optional model ID filter
            n: Number of latest records
        
        Returns:
            List of latest feature importance records
        """
        return self.query_feature_importance(symbol=symbol, model_id=model_id, limit=n)
    
    def get_feature_importance_history(self, feature_name: str, symbol: Optional[str] = None,
                                      days: int = 90) -> List[Dict]:
        """
        Get feature importance history for a specific feature.
        
        Args:
            feature_name: Name of feature
            symbol: Optional symbol filter
            days: Number of days to look back
        
        Returns:
            List of historical importance records
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        return self.query_feature_importance(
            feature_name=feature_name,
            symbol=symbol,
            start_date=start_date
        )
    
    def delete_old_feature_importance(self, days: int = 90) -> int:
        """
        Delete old feature importance records.
        
        Args:
            days: Delete records older than this many days
        
        Returns:
            Number of records deleted
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        query = "DELETE FROM feature_importance WHERE timestamp < ?"
        
        with self.get_connection() as conn:
            result = conn.exec_driver_sql(query, (cutoff_date,))
            deleted_count = result.rowcount
            in_transaction = hasattr(self._transaction_local, 'connection') and self._transaction_local.connection is not None
            if not in_transaction:
                conn.commit()
        
        self.logger.info(f"Deleted {deleted_count} old feature importance records")
        return deleted_count
    
    # Database Maintenance and Backup
    
    def create_backup(self, backup_path: str = None) -> str:
        """Create database backup."""
        if not self.config.get('database.backup_enabled', True):
            self.logger.debug("Backups disabled in config")
            return None
        
        if backup_path is None:
            backup_dir = get_writable_app_dir('backups')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"trading_bot_backup_{timestamp}.db"
            backup_path = os.path.join(backup_dir, backup_filename)
        
        ensure_dir(os.path.dirname(backup_path))
        
        try:
            with self.get_connection() as conn:
                # Access raw sqlite3 connection from SQLAlchemy connection
                # SQLAlchemy Connection wraps DBAPI connection, which wraps sqlite3.Connection
                # For SQLite: connection.connection.connection gives us the actual sqlite3.Connection
                raw_conn = conn.connection.connection
                
                # Use backup API for SQLite
                backup_conn = sqlite3.connect(backup_path)
                raw_conn.backup(backup_conn)
                backup_conn.close()
            
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            raise DatabaseError(
                f"Backup creation failed: {str(e)}",
                details={'backup_path': backup_path, 'error': str(e)}
            )
    
    def vacuum_database(self):
        """Optimize database and reclaim space."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("VACUUM"))
                conn.commit()
            self.logger.info("Database vacuum completed")
        except Exception as e:
            self.logger.error(f"Vacuum failed: {str(e)}")
            raise DatabaseError(f"Vacuum failed: {str(e)}")
    
    def analyze_database(self):
        """Update query optimizer statistics."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("ANALYZE"))
                conn.commit()
            self.logger.info("Database analysis completed")
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise DatabaseError(f"Analysis failed: {str(e)}")
    
    def check_integrity(self) -> bool:
        """Check database integrity."""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                integrity_ok = result[0] == 'ok'
                
                if integrity_ok:
                    self.logger.debug("Database integrity check passed")
                else:
                    self.logger.warning(f"Database integrity check failed: {result[0]}")
                
                return integrity_ok
        except Exception as e:
            self.logger.error(f"Integrity check failed: {str(e)}")
            return False
    
    def get_database_size(self) -> int:
        """Get database file size in bytes."""
        try:
            return os.path.getsize(self.db_path)
        except Exception as e:
            self.logger.error(f"Failed to get database size: {str(e)}")
            return 0
    
    def get_table_row_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        tables = ['trades', 'positions', 'portfolio_snapshots', 'performance_metrics', 'feature_importance']
        counts = {}
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            results = self._execute_query(query, fetch=True)
            counts[table] = results[0]['count'] if results else 0
        
        return counts
    
    def cleanup_old_data(self, retention_days: int = None) -> int:
        """Delete data older than retention period."""
        if retention_days is None:
            retention_days = self.config.get('data.retention_days', 365)
        
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        deleted = 0
        
        # Delete old trades
        deleted += self.delete_trades_before(cutoff_date)
        
        # Delete old snapshots
        deleted += self.delete_snapshots_before(cutoff_date)
        
        # Delete old metrics
        deleted += self.delete_metrics_before(cutoff_date)
        
        # Positions are kept (current positions only)
        
        self.logger.info(f"Data cleanup completed (retention: {retention_days} days, deleted: {deleted} rows)")
        return deleted
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        row_counts = self.get_table_row_counts()
        
        # Get oldest and newest trade dates
        oldest_query = "SELECT MIN(entry_time) as oldest FROM trades"
        newest_query = "SELECT MAX(exit_time) as newest FROM trades"
        
        oldest_result = self._execute_query(oldest_query, fetch=True)
        newest_result = self._execute_query(newest_query, fetch=True)
        
        oldest_date = oldest_result[0]['oldest'] if oldest_result and oldest_result[0]['oldest'] else None
        newest_date = newest_result[0]['newest'] if newest_result and newest_result[0]['newest'] else None
        
        return {
            'total_trades': row_counts.get('trades', 0),
            'total_positions': row_counts.get('positions', 0),
            'oldest_trade_date': oldest_date,
            'newest_trade_date': newest_date,
            'database_size': self.get_database_size(),
            'last_backup_time': None  # Would need to track this
        }
    
    def validate_schema(self) -> bool:
        """Verify all required tables and indexes exist."""
        required_tables = ['trades', 'positions', 'portfolio_snapshots', 'performance_metrics', 'feature_importance']
        
        try:
            with self.get_connection() as conn:
                for table in required_tables:
                    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                    result = conn.execute(text(query)).fetchone()
                    if not result:
                        self.logger.error(f"Missing table: {table}")
                        return False
                
                self.logger.debug("Schema validation passed")
                return True
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            return False
    
    def close(self):
        """
        Close all database connections in the pool.
        
        This disposes the SQLAlchemy engine and closes all connections
        in the connection pool, regardless of which thread they were used in.
        """
        try:
            self.engine.dispose()
            self.logger.info("Database connection pool disposed")
        except Exception as e:
            self.logger.error(f"Error disposing connection pool: {str(e)}")

