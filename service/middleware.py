"""FastAPI middleware for request/response logging."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"status={response.status_code} "
                f"duration={process_time:.3f}s"
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"duration={process_time:.3f}s "
                f"error={str(e)}",
                exc_info=True
            )
            raise
