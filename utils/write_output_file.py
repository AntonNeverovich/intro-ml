def write_answer(file_name: str, answer: str):
    name = './answers/' + file_name
    file = open(name, 'w')
    file.write(answer)
    file.close()
