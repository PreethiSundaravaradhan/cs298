import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate('google-services.json')
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://trialguesture.firebaseio.com'
})

from firebase_admin import db
root = db.reference()

# Obtain a new reference to the user, and retrieve child data.
# Result will be made available as a Python dict.
mary = root.get()


put_out = root.post