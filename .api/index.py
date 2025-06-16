from web_app.app import app as application
import os

# Serve root files
@application.route('/classes.txt')
def classes():
    return application.send_static_file('../../classes.txt')

@application.route('/food_recognition_model.h5')
def model():
    return application.send_static_file('../../food_recognition_model.h5')

# Vercel handler
# Vercel handler
def handler(event, context):
    with app.app_context():
        response = app.full_dispatch_request()
        return response