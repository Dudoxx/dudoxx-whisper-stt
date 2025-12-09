# Apache2 Configuration for Voxtral ASR

This directory contains Apache2 configuration templates for proxying Voxtral ASR streaming server.

## Files

- `voxtral.example.com.conf` - HTTP config with HTTPS redirect
- `voxtral.example.com-ssl.conf` - HTTPS config with WebSocket proxy support

## Installation

1. Copy configs to Apache sites-available:
   ```bash
   sudo cp voxtral.example.com*.conf /etc/apache2/sites-available/
   ```

2. Edit the configs to replace `voxtral.example.com` with your domain:
   ```bash
   sudo sed -i 's/voxtral.example.com/your-domain.com/g' /etc/apache2/sites-available/voxtral.*.conf
   ```

3. Enable required Apache modules:
   ```bash
   sudo a2enmod ssl proxy proxy_http proxy_wstunnel rewrite headers
   ```

4. Get SSL certificate with Let's Encrypt:
   ```bash
   sudo certbot certonly --apache -d your-domain.com
   ```

5. Enable the sites:
   ```bash
   sudo a2ensite voxtral.your-domain.com.conf
   sudo a2ensite voxtral.your-domain.com-ssl.conf
   ```

6. Reload Apache:
   ```bash
   sudo systemctl reload apache2
   ```

## Configuration Details

### WebSocket Support
The SSL config includes WebSocket proxy support for the `/asr` endpoint, enabling real-time audio streaming:
- `mod_proxy_wstunnel` handles WebSocket upgrade
- `ProxyPass /asr ws://localhost:4302/asr` for WebSocket connections

### Proxy Settings
- Backend server: `localhost:4302` (Voxtral streaming server)
- Timeout: 300 seconds (5 minutes for long audio processing)
- CORS headers enabled for cross-origin requests

### Security Headers
Includes standard security headers:
- HSTS (HTTP Strict Transport Security)
- X-Content-Type-Options: nosniff
- X-Frame-Options: SAMEORIGIN
- Referrer-Policy: strict-origin-when-cross-origin
