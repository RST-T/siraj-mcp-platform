"""
SIRAJ v6.1 Production Security Configuration
Comprehensive security implementation for the commercial platform
"""

import os
import secrets
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import wraps
import ipaddress
import re

import jwt
from jwt import PyJWKClient
import redis
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import httpx
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

# Configure security logger
security_logger = logging.getLogger('siraj.security')
security_logger.setLevel(logging.INFO)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # JWT Configuration
    jwt_algorithm: str = 'RS256'
    jwt_audience: str = os.getenv('AUTH0_API_AUDIENCE', 'https://api.siraj.linguistics.org')
    jwt_issuer: str = os.getenv('AUTH0_DOMAIN', 'https://siraj.us.auth0.com/')
    jwks_uri: str = os.getenv('AUTH0_JWKS_URI', 'https://siraj.us.auth0.com/.well-known/jwks.json')
    jwt_leeway: int = 30  # seconds
    
    # API Key Configuration
    api_key_prefix: str = 'siraj_'
    api_key_length: int = 32
    api_key_hash_algorithm: str = 'sha256'
    
    # Rate Limiting
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 100
    rate_limit_burst_multiplier: float = 1.5
    
    # Content Security
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ['.txt', '.md', '.json', '.csv']
    max_source_links: int = 20
    
    # Cultural Sovereignty Protection
    sensitive_content_patterns: List[str] = [
        r'\b(sacred|ritual|ceremonial)\b',
        r'\b(traditional knowledge|indigenous)\b',
        r'\b(spiritual|religious secrets)\b'
    ]
    
    # IP Security
    allowed_countries: List[str] = ['US', 'CA', 'GB', 'AU', 'DE', 'FR']  # Academic-friendly countries
    blocked_ip_ranges: List[str] = []  # Can be populated with known threat IPs
    
    # Encryption
    encryption_key: bytes = Fernet.generate_key() if not os.getenv('ENCRYPTION_KEY') else os.getenv('ENCRYPTION_KEY').encode()

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, config: SecurityConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.jwks_client = PyJWKClient(config.jwks_uri)
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.cipher_suite = Fernet(config.encryption_key)
        
        # Compile regex patterns for performance
        self.sensitive_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in config.sensitive_content_patterns]
    
    def generate_api_key(self, user_id: str) -> tuple[str, str]:
        """Generate a new API key and return (key, hash)"""
        random_part = secrets.token_urlsafe(self.config.api_key_length)
        api_key = f"{self.config.api_key_prefix}{random_part}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Log key generation (without exposing the key)
        security_logger.info(f"API key generated for user {user_id}", extra={
            'user_id': user_id,
            'key_hash': key_hash[:8] + '...',
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return api_key, key_hash
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload"""
        try:
            # Get signing key
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[self.config.jwt_algorithm],
                audience=self.config.jwt_audience,
                issuer=self.config.jwt_issuer,
                leeway=self.config.jwt_leeway
            )
            
            # Additional validation
            self._validate_token_claims(payload)
            
            security_logger.info("JWT token validated successfully", extra={
                'user_id': payload.get('sub', 'unknown'),
                'token_exp': payload.get('exp'),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.warning("JWT token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTClaimsError as e:
            security_logger.warning(f"JWT claims validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid token claims")
        except Exception as e:
            security_logger.error(f"JWT validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def _validate_token_claims(self, payload: Dict[str, Any]):
        """Validate additional JWT claims"""
        required_claims = ['sub', 'aud', 'iss', 'exp']
        for claim in required_claims:
            if claim not in payload:
                raise jwt.JWTClaimsError(f"Missing required claim: {claim}")
        
        # Validate custom claims
        if 'scope' in payload:
            valid_scopes = ['read', 'write', 'admin']
            scopes = payload['scope'].split()
            for scope in scopes:
                if scope not in valid_scopes:
                    raise jwt.JWTClaimsError(f"Invalid scope: {scope}")
    
    def rate_limit_check(self, identifier: str, limit: Optional[int] = None) -> bool:
        """Check rate limiting for identifier (user_id, IP, etc.)"""
        limit = limit or self.config.rate_limit_max_requests
        window = self.config.rate_limit_window
        
        # Use sliding window rate limiting
        now = int(time.time())
        pipeline = self.redis.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(f"rate_limit:{identifier}", 0, now - window)
        
        # Count current requests
        pipeline.zcard(f"rate_limit:{identifier}")
        
        # Add current request
        pipeline.zadd(f"rate_limit:{identifier}", {str(now): now})
        pipeline.expire(f"rate_limit:{identifier}", window)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        if current_requests >= limit:
            security_logger.warning(f"Rate limit exceeded for {identifier}", extra={
                'identifier': identifier,
                'current_requests': current_requests,
                'limit': limit,
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
        
        return True
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address against security policies"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check against blocked ranges
            for blocked_range in self.config.blocked_ip_ranges:
                if ip_obj in ipaddress.ip_network(blocked_range):
                    security_logger.warning(f"Blocked IP attempted access: {ip}")
                    return False
            
            # Additional checks for private/reserved IPs in production
            if ip_obj.is_private and os.getenv('ENVIRONMENT') == 'production':
                security_logger.info(f"Private IP access in production: {ip}")
                # Allow for now, but log for monitoring
            
            return True
            
        except ValueError:
            security_logger.error(f"Invalid IP address format: {ip}")
            return False
    
    def scan_for_sensitive_content(self, content: str) -> List[str]:
        """Scan content for culturally sensitive patterns"""
        detected_patterns = []
        
        for pattern in self.sensitive_patterns:
            matches = pattern.findall(content)
            if matches:
                detected_patterns.extend(matches)
        
        if detected_patterns:
            security_logger.info(f"Sensitive content detected: {len(detected_patterns)} patterns", extra={
                'pattern_count': len(detected_patterns),
                'content_length': len(content),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return detected_patterns
    
    def validate_source_citations(self, sources: List[Dict[str, Any]]) -> bool:
        """Validate source citations for academic integrity"""
        if len(sources) > self.config.max_source_links:
            security_logger.warning(f"Too many source citations: {len(sources)}")
            return False
        
        required_fields = ['title', 'author', 'url']
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for source in sources:
            # Check required fields
            for field in required_fields:
                if field not in source or not source[field]:
                    security_logger.warning(f"Missing required field in source: {field}")
                    return False
            
            # Validate URL format
            if not url_pattern.match(source['url']):
                security_logger.warning(f"Invalid URL in source: {source['url']}")
                return False
            
            # Check for suspicious domains (basic implementation)
            suspicious_domains = ['bit.ly', 'tinyurl.com', 't.co']
            for domain in suspicious_domains:
                if domain in source['url']:
                    security_logger.warning(f"Suspicious domain in source: {domain}")
                    return False
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def generate_csrf_token(self, user_id: str) -> str:
        """Generate CSRF token for forms"""
        timestamp = int(time.time())
        message = f"{user_id}:{timestamp}"
        signature = hmac.new(
            self.config.encryption_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{message}:{signature}"
    
    def validate_csrf_token(self, token: str, user_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            token_user_id, timestamp_str, signature = parts
            
            # Check user ID
            if token_user_id != user_id:
                return False
            
            # Check timestamp
            timestamp = int(timestamp_str)
            if int(time.time()) - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{token_user_id}:{timestamp_str}"
            expected_signature = hmac.new(
                self.config.encryption_key,
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, TypeError):
            return False

class SecurityMiddleware:
    """FastAPI middleware for security enforcement"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
    
    async def __call__(self, request: Request, call_next):
        """Process security checks for each request"""
        start_time = time.time()
        
        try:
            # IP validation
            client_ip = self._get_client_ip(request)
            if not self.security.validate_ip_address(client_ip):
                raise HTTPException(status_code=403, detail="Access denied from this IP")
            
            # Rate limiting
            rate_limit_key = f"ip:{client_ip}"
            if not self.security.rate_limit_check(rate_limit_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Content length check
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > self.security.config.max_content_length:
                raise HTTPException(status_code=413, detail="Request entity too large")
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers.update({
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
                'Referrer-Policy': 'strict-origin-when-cross-origin'
            })
            
            # Log successful request
            processing_time = time.time() - start_time
            security_logger.info(f"Request processed successfully", extra={
                'client_ip': client_ip,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            security_logger.error(f"Security middleware error: {e}", extra={
                'client_ip': self._get_client_ip(request),
                'method': request.method,
                'path': request.url.path,
                'timestamp': datetime.utcnow().isoformat()
            })
            raise HTTPException(status_code=500, detail="Internal security error")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else '0.0.0.0'

def security_required(scopes: List[str] = None):
    """Decorator for endpoints requiring authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and security dependencies
            request = None
            security_manager = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                elif isinstance(arg, SecurityManager):
                    security_manager = arg
            
            if not request or not security_manager:
                raise HTTPException(status_code=500, detail="Security configuration error")
            
            # Validate authentication
            auth_header = request.headers.get('authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            
            try:
                payload = security_manager.validate_jwt_token(token)
                
                # Check scopes if required
                if scopes:
                    token_scopes = payload.get('scope', '').split()
                    if not any(scope in token_scopes for scope in scopes):
                        raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Add user context to kwargs
                kwargs['current_user'] = payload
                
                return await func(*args, **kwargs)
                
            except HTTPException:
                raise
            except Exception as e:
                security_logger.error(f"Authentication failed: {e}")
                raise HTTPException(status_code=401, detail="Authentication failed")
        
        return wrapper
    return decorator

# Initialize global security manager
def create_security_manager() -> SecurityManager:
    """Factory function to create security manager"""
    config = SecurityConfig()
    
    # Initialize Redis client
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    
    return SecurityManager(config, redis_client)

# Security monitoring and alerting
class SecurityMonitor:
    """Monitor security events and send alerts"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.alert_thresholds = {
            'failed_logins': 10,
            'rate_limit_violations': 50,
            'sensitive_content_detections': 5
        }
    
    async def check_threat_indicators(self):
        """Check for threat indicators and send alerts"""
        # Implementation would include:
        # - Monitoring failed authentication attempts
        # - Detecting brute force attacks
        # - Analyzing access patterns
        # - Sending alerts to security team
        pass
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate security metrics report"""
        # Implementation would analyze security logs
        # and generate comprehensive security metrics
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'threat_level': 'low',
            'incidents': 0,
            'blocked_requests': 0,
            'active_users': 0
        }