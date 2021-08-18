from __init__ import *
import sqlite3

'''
all functions and query used to connect with the sqlite db
- create tables
- insert data to the tables
- create connection
'''

# papers table
create_papers_table = '''
CREATE TABLE IF NOT EXISTS papers (
	paper_id text PRIMARY KEY,
	title text,
    paper_path text,
    doi text,
    date text,
    authors text,
    affiliations text
);
'''

create_paragraph_table = '''
CREATE TABLE IF NOT EXISTS paragraph (
    paragraph_id text PRIMARY KEY,
    body_text text NOT NULL,
    paper_id text NOT NULL,
    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
)
'''

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(e)

def add_paper(conn, paper_dict):
    """
    Add a new paper into the paper table
    :param conn:
    :param paper:
    """
    #(paper_id, title, paper_path, doi, date, authors, affiliations, paragraphs_candidates)
    sql = ''' INSERT INTO papers(paper_id, title, paper_path, doi, date, authors, affiliations)
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tuple(paper_dict.values()))
    conn.commit()
    return

def add_paragraph(conn, para):
    """
    Add a new paper into the paper table
    :param conn:
    :param paper:
    """
    #(paper_id, title, paper_path, doi, date, authors, affiliations, paragraphs_candidates)
    sql = ''' INSERT INTO paragraph(paragraph_id, paper_id, body_text)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tuple(para.values()))
    conn.commit()
    return