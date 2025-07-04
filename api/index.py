from flask import Flask
from main_app import app  # This imports your original app

# This creates a Vercel-compatible WSGI application
vercel_app = app

# For local testing
if __name__ == '__main__':
    app.run()