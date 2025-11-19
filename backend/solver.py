def sudokoSolver(mat):
    rHash=[set() for _ in range(9)]
    cHash=[set() for _ in range(9)]
    boxHash=[set() for _ in range(9)]

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            rHash[i].add(mat[i][j])
            cHash[j].add(mat[i][j])
            boxHash[(i//3)*3+j//3].add(mat[i][j])

    def backtrack(r,c,board):
        if r==9:
            return True
        if c==9:
            return backtrack(r+1,0,board)
        if board[r][c]!="0":
            return backtrack(r,c+1,board)
        for idx in map(str,range(1,10)):
            bidx=(r//3)*3+c//3
            if board[r][c]=="0" and idx not in rHash[r] and idx not in cHash[c] and idx not in boxHash[bidx]:
                board[r][c]=idx
                rHash[r].add(idx)
                cHash[c].add(idx)

                if backtrack(r,c,board):
                    return True
                board[r][c]="0"
                rHash[r].remove(idx)
                cHash[c].remove(idx)
        return False
    if backtrack(0,0,mat):
        return mat
    else:
        return [[]]

''' To be completed!'''
def extractImage(image):
    pass