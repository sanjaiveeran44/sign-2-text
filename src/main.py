from data_collection import collect_sign_data

if __name__ == "__main__":
    actions = ['hello', 'thank you', 'yes', 'no']
    collect_sign_data(actions, num_sequences=30, sequence_length=30)
