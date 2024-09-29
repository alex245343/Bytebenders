import os
import sqlite3

current_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_directory, 'Simvoly', 'osnov')

db_path = os.path.join(current_directory, 'Simvoly', 'osnov.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    file_path TEXT,
    group_name TEXT
)
''')

def get_group_name(filename):
    if filename.startswith('1_'):
        return 'Группировка АУЕ'
    elif filename.startswith('2_'):
        return 'Символика Третьего Рейха'
    elif filename.startswith('3_'):
        return 'Партия НБП'
    elif filename.startswith('4_'):
        return 'Русский Общенациональный Союз'
    elif filename.startswith('5_'):
        return 'ДПНИ'
    elif filename.startswith('6_'):
        return 'ЛГБТ'
    elif filename.startswith('7_'):
        return 'Лада'
    elif filename.startswith('8_'):
        return 'Блогер, Дмитрий Пучков'
    return 'Неизвестная группа'

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        file_path = os.path.join(folder_path, filename)
        group_name = get_group_name(filename)

        cursor.execute('INSERT INTO images (filename, file_path, group_name) VALUES (?, ?, ?)', 
                       (filename, file_path, group_name))

conn.commit()
conn.close()
