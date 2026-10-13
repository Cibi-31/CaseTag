# ============================================================
# CASETag - ROLE BASED AUTHENTICATION
# ============================================================

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(
    BASE_DIR,
    "casetag_users.db"
)

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "casetag-demo-secret-change-this"
)

JWT_ALGORITHM = "HS256"

TOKEN_EXPIRE_HOURS = 8


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            salt TEXT NOT NULL,

            role TEXT NOT NULL,

            created_at TEXT NOT NULL
        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return password_hash, salt


def verify_password(
    password,
    password_hash,
    salt
):

    calculated_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return secrets.compare_digest(
        calculated_hash,
        password_hash
    )


# ============================================================
# CREATE USER
# ============================================================

def create_user(
    name,
    email,
    password,
    role="common"
):

    password_hash, salt = hash_password(
        password
    )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password_hash,
                salt,
                role,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                email.lower().strip(),
                password_hash,
                salt,
                role,
                datetime.now(timezone.utc).isoformat()
            )
        )

        connection.commit()

        user_id = cursor.lastrowid

        return user_id

    except sqlite3.IntegrityError:

        return None

    finally:

        connection.close()


# ============================================================
# FIND USER
# ============================================================

def get_user_by_email(email):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (
            email.lower().strip(),
        )
    )

    user = cursor.fetchone()

    connection.close()

    return user


# ============================================================
# AUTHENTICATE
# ============================================================

def authenticate_user(
    email,
    password
):

    user = get_user_by_email(
        email
    )

    if not user:

        return None

    valid = verify_password(
        password,
        user["password_hash"],
        user["salt"]
    )

    if not valid:

        return None

    return user


# ============================================================
# JWT
# ============================================================

def create_access_token(user):

    expire = (
        datetime.now(timezone.utc)
        +
        timedelta(
            hours=TOKEN_EXPIRE_HOURS
        )
    )

    payload = {

        "sub": str(user["id"]),

        "name": user["name"],

        "email": user["email"],

        "role": user["role"],

        "exp": expire
    }

    token = jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )

    return token


# ============================================================
# VERIFY TOKEN
# ============================================================

def decode_token(token):

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[
                JWT_ALGORITHM
            ]
        )

        return payload

    except jwt.ExpiredSignatureError:

        return None

    except jwt.InvalidTokenError:

        return None


# ============================================================
# INITIALIZE
# ============================================================

init_database()
def create_default_professional_accounts():

    # --------------------------------------------------------
    # DEMO LAWYER
    # --------------------------------------------------------

    lawyer_email = os.getenv(
        "LAWYER_EMAIL",
        "lawyer@casetag.demo"
    )

    lawyer_password = os.getenv(
        "LAWYER_PASSWORD",
        "Lawyer@123"
    )

    if not get_user_by_email(
        lawyer_email
    ):

        create_user(
            name="CASETAG Lawyer",
            email=lawyer_email,
            password=lawyer_password,
            role="lawyer"
        )

    # --------------------------------------------------------
    # DEMO JUDGE
    # --------------------------------------------------------

    judge_email = os.getenv(
        "JUDGE_EMAIL",
        "judge@casetag.demo"
    )

    judge_password = os.getenv(
        "JUDGE_PASSWORD",
        "Judge@123"
    )

    if not get_user_by_email(
        judge_email
    ):

        create_user(
            name="CASETAG Judge",
            email=judge_email,
            password=judge_password,
            role="judge"
        )


create_default_professional_accounts()