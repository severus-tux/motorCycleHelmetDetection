#!/usr/bin/python
 
import urllib.request
import urllib.parse
import sys
 
def sendSMS(apikey, numbers, sender, message):
    data =  urllib.parse.urlencode({'apikey': apikey, 'numbers': numbers,
        'message' : message, 'sender': sender})
    data = data.encode('utf-8')
    request = urllib.request.Request("https://api.textlocal.in/send/?")
    f = urllib.request.urlopen(request, data)
    fr = f.read()
    return(fr)


message = (" ".join(sys.argv[1:]))

resp =  sendSMS('+xHt9RJlU1Y-sJB8P6D5jMzuU00wPOXBa68OdEr8FD', '917019441892',
    'TXTLCL', message)
#print (resp)
