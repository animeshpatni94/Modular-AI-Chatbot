import pyodbc
import configparser

def get_db_connection():
    config = configparser.ConfigParser()
    config.read('config.ini')
    provider = config['DB'].get('provider', 'sqlserver').lower()
    if provider == 'sqlserver':
        conn_str = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={config['DB']['server']};"
            f"DATABASE={config['DB']['database']};"
            f"UID={config['DB']['user']};"
            f"PWD={config['DB']['password']};"
            "TrustServerCertificate=Yes"
        )
    elif provider == 'postgresql':
        conn_str = (
            f"DRIVER={{PostgreSQL Unicode}};"
            f"SERVER={config['DB']['server']};"
            f"DATABASE={config['DB']['database']};"
            f"UID={config['DB']['user']};"
            f"PWD={config['DB']['password']};"
            f"PORT={config['DB'].get('port', 5432)}"
        )
    else:
        raise ValueError(f"Unknown DB provider: {provider}")
    return pyodbc.connect(conn_str)
