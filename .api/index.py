from web_app.app import app as application
import os

# Vercel handler function
def handler(event, context):
    # Set environment variables if needed
    os.environ['VERCEL'] = '1'
    
    # Process the event through Flask
    with application.app_context():
        response = application.full_dispatch_request()
        
        # Convert Flask response to Vercel format
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True)
        }

# For local testing (if needed)
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080)