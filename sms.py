from kavenegar import *
api = KavenegarAPI('6A3644686D4F4864644F74736A415844564F413834596D4D4B5748527544377951586159746263477465453D')
params = { 'sender' : '2000500666', 'receptor': '09335858978', 'message' :'کاربر' }
api.sms_send( params)
        