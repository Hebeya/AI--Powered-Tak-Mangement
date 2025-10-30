from flask import Flask, jsonify, request
import pandas as pd
from sqlalchemy import create_engine

app = Flask(__name__)
engine = create_engine("mysql+pymysql://root:HS%402223@localhost:3306/task_management")

@app.route('/tasks', methods=['GET'])
def get_tasks():
    df = pd.read_sql("SELECT * FROM tasks", engine)
    return jsonify(df.to_dict(orient='records'))

@app.route('/tasks/<int:id>', methods=['GET'])
def get_task(id):
    df = pd.read_sql(f"SELECT * FROM tasks WHERE task_id={id}", engine)
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
