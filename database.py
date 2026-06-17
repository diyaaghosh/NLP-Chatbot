import sqlite3
import hashlib
from datetime import datetime


conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

cursor = conn.cursor()


# Users table

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")


# Chat table

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    role TEXT,
    message TEXT,
    time TEXT
)
""")


conn.commit()



def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()



def register(username,password):

    try:

        cursor.execute(
        """
        INSERT INTO users(username,password)
        VALUES(?,?)
        """,
        (
            username,
            hash_password(password)
        )
        )

        conn.commit()

        return True


    except:

        return False




def login(username,password):

    cursor.execute(
    """
    SELECT id
    FROM users
    WHERE username=?
    AND password=?
    """,
    (
        username,
        hash_password(password)
    )
    )


    result = cursor.fetchone()


    if result:

        return result[0]

    return None





def save_chat(user_id,role,message):

    cursor.execute(
    """
    INSERT INTO chats
    (user_id,role,message,time)
    VALUES(?,?,?,?)
    """,
    (
        user_id,
        role,
        message,
        datetime.now()
    )
    )


    conn.commit()




def load_chats(user_id):

    cursor.execute(
    """
    SELECT role,message
    FROM chats
    WHERE user_id=?
    ORDER BY id
    """,
    (user_id,)
    )


    rows = cursor.fetchall()


    return [
        {
        "role":r[0],
        "content":r[1]
        }
        for r in rows
    ]





def delete_chats(user_id):

    cursor.execute(
    """
    DELETE FROM chats
    WHERE user_id=?
    """,
    (user_id,)
    )


    conn.commit()