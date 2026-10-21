import sqlite3
import random
from datetime import datetime, timedelta

conn = sqlite3.connect("bank.db")
cursor = conn.cursor()

# Drop old tables
cursor.executescript("""
                     DROP TABLE IF EXISTS customers;
                     DROP TABLE IF EXISTS addresses;
                     DROP TABLE IF EXISTS branches;
                     DROP TABLE IF EXISTS accounts;
                     DROP TABLE IF EXISTS transactions;
                     DROP TABLE IF EXISTS loans;
                     DROP TABLE IF EXISTS employees;
                     DROP TABLE IF EXISTS account_managers;
                     """)

# Customers
cursor.execute("""
               CREATE TABLE customers
               (
                   customer_id INTEGER PRIMARY KEY,
                   name        TEXT,
                   email       TEXT,
                   phone       TEXT,
                   join_date   DATE
               )
               """)

# Addresses
cursor.execute("""
               CREATE TABLE addresses
               (
                   address_id  INTEGER PRIMARY KEY,
                   customer_id INTEGER,
                   city        TEXT,
                   state       TEXT,
                   country     TEXT,
                   FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
               )
               """)

# Branches
cursor.execute("""
               CREATE TABLE branches
               (
                   branch_id   INTEGER PRIMARY KEY,
                   branch_name TEXT,
                   city        TEXT
               )
               """)

# Accounts
cursor.execute("""
               CREATE TABLE accounts
               (
                   account_id   INTEGER PRIMARY KEY,
                   customer_id  INTEGER,
                   branch_id    INTEGER,
                   account_type TEXT,
                   balance      REAL,
                   open_date    DATE,
                   FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
                   FOREIGN KEY (branch_id) REFERENCES branches (branch_id)
               )
               """)

# Transactions
cursor.execute("""
               CREATE TABLE transactions
               (
                   txn_id     INTEGER PRIMARY KEY,
                   account_id INTEGER,
                   txn_date   DATE,
                   amount     REAL,
                   txn_type   TEXT,
                   FOREIGN KEY (account_id) REFERENCES accounts (account_id)
               )
               """)

# Loans
cursor.execute("""
               CREATE TABLE loans
               (
                   loan_id       INTEGER PRIMARY KEY,
                   customer_id   INTEGER,
                   loan_type     TEXT,
                   amount        REAL,
                   interest_rate REAL,
                   status        TEXT,
                   FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
               )
               """)

# Employees
cursor.execute("""
               CREATE TABLE employees
               (
                   employee_id INTEGER PRIMARY KEY,
                   name        TEXT,
                   role        TEXT,
                   branch_id   INTEGER,
                   FOREIGN KEY (branch_id) REFERENCES branches (branch_id)
               )
               """)

# Account Managers
cursor.execute("""
               CREATE TABLE account_managers
               (
                   manager_id INTEGER,
                   account_id INTEGER,
                   PRIMARY KEY (manager_id, account_id),
                   FOREIGN KEY (manager_id) REFERENCES employees (employee_id),
                   FOREIGN KEY (account_id) REFERENCES accounts (account_id)
               )
               """)

branches = [
    (1, "Dhaka Main", "Dhaka"),
    (2, "Basabo Branch", "Dhaka/Basabo"),
    (3, "Feni Main", "Feni")
]
cursor.executemany("INSERT INTO branches VALUES (?,?,?)", branches)

names = ["Minhaz Uddin", "Mr x", "Mr y", "Mr z"]

for i in range(1, 51):
    name = random.choice(names) + str(i)
    cursor.execute("""
                   INSERT INTO customers
                   VALUES (?, ?, ?, ?, ?)
                   """, (
                       i,
                       name,
                       f"{name.lower()}@mail.com",
                       f"98{random.randint(10000000, 99999999)}",
                       datetime.now() - timedelta(days=random.randint(100, 2000))
                   ))

cities = ["Dhaka", "Feni"]

for i in range(1, 51):
    cursor.execute("""
                   INSERT INTO addresses
                   VALUES (?, ?, ?, ?, ?)
                   """, (
                       i,
                       i,
                       random.choice(cities),
                       "Feni",
                       "Dhaka"
                   ))

account_types = ["Savings", "Current", "Fixed"]

for i in range(1, 101):
    cursor.execute("""
                   INSERT INTO accounts
                   VALUES (?, ?, ?, ?, ?, ?)
                   """, (
                       i,
                       random.randint(1, 50),
                       random.randint(1, 3),
                       random.choice(account_types),
                       random.randint(1000, 500000),
                       datetime.now() - timedelta(days=random.randint(10, 1500))
                   ))

txn_types = ["Deposit", "Withdrawal"]

txn_id = 1
for acc_id in range(1, 101):
    for _ in range(random.randint(20, 50)):
        cursor.execute("""
                       INSERT INTO transactions
                       VALUES (?, ?, ?, ?, ?)
                       """, (
                           txn_id,
                           acc_id,
                           datetime.now() - timedelta(days=random.randint(1, 365)),
                           random.randint(100, 20000),
                           random.choice(txn_types)
                       ))
        txn_id += 1

loan_types = ["Home", "Personal", "Auto"]

for i in range(1, 30):
    cursor.execute("""
                   INSERT INTO loans
                   VALUES (?, ?, ?, ?, ?, ?)
                   """, (
                       i,
                       random.randint(1, 50),
                       random.choice(loan_types),
                       random.randint(50000, 1000000),
                       round(random.uniform(5, 12), 2),
                       random.choice(["Active", "Closed"])
                   ))

for i in range(1, 11):
    cursor.execute("""
                   INSERT INTO employees
                   VALUES (?, ?, ?, ?)
                   """, (
                       i,
                       f"Employee{i}",
                       "Manager",
                       random.randint(1, 3)
                   ))

for acc in range(1, 101):
    cursor.execute("""
                   INSERT INTO account_managers
                   VALUES (?, ?)
                   """, (
                       random.randint(1, 10),
                       acc
                   ))

conn.commit()
conn.close()

print("✅ Realistic Banking Database Created!")
