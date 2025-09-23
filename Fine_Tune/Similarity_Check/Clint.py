import requests

while True:
    prompt = input("Hi there! how can I help you? : ")
    if prompt.lower() == 'quit':
            break
        
    payload = {'prompt' : prompt}
    response = requests.post(url = 'http://127.0.0.1:8000/generate', json = payload)

    if response.status_code == 200:
        output_text = response.json()['output_text']
        print(output_text)
        print("Time taken : ",round(response.json()['time_taken'],2))
    else:
        print("Something went wrong",response.status_code)