from flask import Flask, jsonify, abort, request
import pandas as pd

sim_df = pd.read_csv('input.csv', parse_dates=['SETTLEMENTDATE'])
# avoids issues with jsonify datetime
sim_df['SETTLEMENTDATE'] = sim_df['SETTLEMENTDATE'].astype(str)
row_index = 0

post_data = []

app = Flask(__name__)
@app.route('/forecast', methods=['GET'])
def forecast():
    global row_index
    global sim_df
    global post_data

    if row_index >= len(sim_df):
        post_df = pd.DataFrame(post_data)
        post_df.to_csv('output.csv', index=False)

        # reset row_index so can run again
        row_index = 0

        abort(400, 'Data not found')

    row = dict(sim_df.iloc[row_index])
    row_index += 1
    
    return jsonify(row)

@app.route('/action', methods=['POST'])
def action():
    global post_data

    if not request.json:
        abort(400, 'No data received')
    
    data = request.get_json()

    post_data.append(data)

    return jsonify({'status':'Success'}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)