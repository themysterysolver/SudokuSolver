from flask import Flask, render_template, request, jsonify
from solver import sudokoSolver,extractImage
app = Flask(__name__)

@app.route('/solve', methods=['POST'])
def solve():
    '''should complete extractImage function!!
        Make it return as array of strings => 0 being an empty character '''
    #image = request.files['image']
    #board = extractImage(image)

    ''' The below part should be commented out'''
    board = [
        ['0', '7', '0', '0', '0', '0', '0', '4', '3'],
        ['0', '4', '0', '0', '0', '9', '6', '1', '0'],
        ['8', '0', '0', '6', '3', '4', '9', '0', '0'],
        ['0', '9', '4', '0', '5', '2', '0', '0', '0'],
        ['3', '5', '8', '4', '6', '0', '0', '2', '0'],
        ['0', '0', '0', '8', '0', '0', '5', '3', '0'],
        ['0', '8', '0', '0', '7', '0', '0', '9', '1'],
        ['9', '0', '2', '1', '0', '0', '0', '0', '5'],
        ['0', '0', '7', '0', '4', '0', '8', '0', '2']
    ]
    solved_board = sudokoSolver(board)
    return jsonify({"board": solved_board})

if __name__ == '__main__':
    app.run(debug=True)