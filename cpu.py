#!env/bin/python
import math
import numpy as np
from cycler import cycler
import traceback
import types
import random
import time

def create_table(name, d, N, cols):
    data = []
    for c in cols:
        column = range(0,d) * (N/d)
        random.shuffle(column)
        data.append(column)
    data = np.array(data)
    data = np.transpose(data)

    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS {}".format(name))
    args = ','.join(['{} INT'.format(c) for c in cols])
    cursor.execute("CREATE TABLE {} ({})".format(name,args))
    cols = ','.join(['{}'.format(c) for c in cols])
    for row in data:
        vals = ','.join([str(r) for r in row])
        cursor.execute('INSERT INTO {}({}) VALUES({})'.format(name, cols, vals))

def create_database(dbtp):
    global db
    if dbtp == 'mysql':
        import mysql.connector
        db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                passwd = "666"
                )
        cursor = db.cursor(buffered=True)
        cursor.execute("DROP DATABASE IF EXISTS `datacamp`")
        cursor.execute("CREATE DATABASE datacamp")
        db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                passwd = "666",
                database='datacamp'
                )
    elif dbtp == 'psql':
        import psycopg2
        db = psycopg2.connect(
                host = "localhost",
                user = "yaqiz",
                password = "666"
                )
        cursor = db.cursor()
        # cursor.execute("DROP DATABASE IF EXISTS datacamp")
        # cursor.execute("CREATE DATABASE datacamp")
        # db = mysql.connector.connect(
                # host = "localhost",
                # user = "postgres",
                # password = "666",
                # dbname='datacamp'
                # )

def join(d,N):
    create_database(dbtp='psql')
    cursor = db.cursor()
    create_table('R',d,N,['A','B'])
    create_table('S',d,N,['C','D','E'])
    create_table('T',d,N,['F','G'])
    db.commit()
    start = time.time()
    print('starting join N={} d={}'.format(N,d))
    join1 = "select * from R INNER JOIN S on R.B = S.C"
    join2 = "select * from ({}) as RS INNER JOIN T on RS.C = T.F".format(join1)
    count = "select count (A) from ({}) as RST".format(join2)
    cursor.execute(count)
    elapse = time.time()-start
    result = cursor.fetchall()
    cursor.close()
    return elapse

def selfjoin():
    Ns = 100 + np.array(range(1,20,1)) * 500
    # Ns = [100]
    with open('data/cpu.csv', 'w') as f:
        f.write('N,d,cpu_time\n')
        for N in Ns:
            for d in [1,2,4,5,10]:
                d = N/d
                time = join(d,N)
                f.write('{},{},{}\n'.format(N,d,time))
                f.flush()

def main():
    selfjoin()

if __name__ == "__main__":
    main()
