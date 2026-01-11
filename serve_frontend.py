import http.server
import socketserver

PORT = 3000
Handler = http.server.SimpleHTTPRequestHandler

print(f"Frontend serving at http://localhost:{PORT}/index.html")
print("Press Ctrl+C to stop.")

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nFrontend stopped.")
