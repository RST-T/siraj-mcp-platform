#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for SIRAJ MCP Server HTTPS transport
This script creates a self-signed certificate for testing purposes only.
For production, use certificates from a trusted Certificate Authority.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def generate_ssl_certificate():
    """Generate self-signed SSL certificate for HTTPS transport"""
    
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress
    except ImportError:
        print("Missing required dependencies. Please install cryptography:")
        print("   pip install cryptography")
        sys.exit(1)
    
    # Create certs directory
    cert_dir = Path("certs")
    cert_dir.mkdir(exist_ok=True)
    
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"
    
    print("Generating self-signed SSL certificate for SIRAJ MCP Server...")
    print(f"   Certificate: {cert_file}")
    print(f"   Private Key: {key_file}")
    print()
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Create certificate subject
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Development"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SIRAJ Team"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "MCP Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    # Create certificate
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        datetime.utcnow() + timedelta(days=365)  # Valid for 1 year
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv6Address("::1")),
        ]),
        critical=False,
    ).add_extension(
        x509.KeyUsage(
            digital_signature=True,
            key_encipherment=True,
            content_commitment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        critical=True,
    ).add_extension(
        x509.ExtendedKeyUsage([
            x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
        ]),
        critical=True,
    ).sign(private_key, hashes.SHA256())
    
    # Write private key
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    # Set appropriate permissions
    os.chmod(key_file, 0o600)  # Private key should be readable only by owner
    os.chmod(cert_file, 0o644)  # Certificate can be world-readable
    
    print("SSL certificate generated successfully!")
    print()
    print("Usage:")
    print("   # Start SIRAJ MCP server with HTTPS")
    print("   siraj-mcp-server start --transport https --port 3443")
    print()
    print("   # Or with explicit certificate files")
    print(f"   siraj-mcp-server start --transport https --port 3443 \\")
    print(f"       --ssl-cert {cert_file} --ssl-key {key_file}")
    print()
    print("SECURITY WARNING:")
    print("   This is a SELF-SIGNED certificate for DEVELOPMENT/TESTING only!")
    print("   For production, use certificates from a trusted Certificate Authority.")
    print()
    print("Certificate Information:")
    print(f"   Subject: {subject.rfc4514_string()}")
    print(f"   Valid From: {cert.not_valid_before}")
    print(f"   Valid Until: {cert.not_valid_after}")
    print(f"   Serial Number: {cert.serial_number}")
    
    return cert_file, key_file

if __name__ == "__main__":
    try:
        generate_ssl_certificate()
    except KeyboardInterrupt:
        print("\nCertificate generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating certificate: {e}")
        sys.exit(1)