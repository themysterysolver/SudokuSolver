import streamlit as st
import requests

BACKEND_URL="http://127.0.0.1:5000/solve"

''' Generated code here!'''

def render_sudoku(board):
    """Render Sudoku board as HTML table with 3x3 box borders"""
    html = '<table style="border-collapse: collapse;">'
    for i, row in enumerate(board):
        html += '<tr>'
        for j, cell in enumerate(row):
            # Thick borders for 3x3 boxes
            top = '3px' if i % 3 == 0 else '1px'
            left = '3px' if j % 3 == 0 else '1px'
            right = '3px' if j == 8 else '1px'
            bottom = '3px' if i == 8 else '1px'
            
            html += f'<td style="border-top:{top} solid black; border-left:{left} solid black; border-bottom:{bottom} solid black; border-right:{right} solid black; text-align: center; width:40px; height:40px; font-size:20px;">{cell}</td>'
        html += '</tr>'
    html += '</table>'
    return html

st.title("Sudoku Solver")
uploaded_file = st.file_uploader("Upload Sudoku Image", type=["jpg", "jpeg", "png"],accept_multiple_files=False)
if uploaded_file:
    if st.button('Solve'):
        file={"image":uploaded_file.getvalue()}
        res = requests.post(BACKEND_URL,files=file)

        if res.status_code == 200:
            data = res.json()
            board = data.get('board',[])
            # if board:
            #     for i, row in enumerate(board):
            #         cols = st.columns(9)
            #         for j, cell in enumerate(row):
            #             cols[j].write(cell)
            # else:
            #     st.warning('No board is returned!')
            if board:
                st.markdown(render_sudoku(board), unsafe_allow_html=True)
            else:
                st.warning("No board returned!")
        else:
            st.warning(f"{res.status_code}")
                    


