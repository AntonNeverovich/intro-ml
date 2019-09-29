names = ['\"Cumings\"', '(Flo\"rence', ')Tha)yer)', 'Lai((((,""",,na']
names2 = []
names2 = []
print(names)


def delete_symbol(elem, symbol):
    if elem.find(symbol):
        return elem.replace(symbol, '')  # type: str
    else:
        return elem


def cleaning_data1(list, symbol1, symbol2, symbol3, symbol4):
    for i in range(0, len(list)):
        if list[i][0] == symbol1 or list[i][0] == symbol2 or list[i][0] == symbol3:
            list[i] = list[i][1:]

        list[i] = delete_symbol(list[i], symbol1)
        list[i] = delete_symbol(list[i], symbol2)
        list[i] = delete_symbol(list[i], symbol3)
        list[i] = delete_symbol(list[i], symbol4)
    return list


def cleaning_data2(list, symbols):
    for i in range(len(list)):
        for j in range(len(symbols)):
            if list[i][0] == symbols[j]:
                list[i] = list[i][1:]
        list[i] = delete_symbol(list[i], symbols[j])
    return list


names2 = cleaning_data1(names, '\"', '(', ')', ',')

print(names2)
