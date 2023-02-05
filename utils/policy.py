import cvxpy as cp
import numpy as np
import gurobipy
import pandas as pd

def gurobipy_sol_ILP(NUM_JOB, NUM_RESOURCE, NUM_DB,
                matrix_t, matrix_p, alpha):
    # IS OK!
    job_indexes = matrix_t.index
    resource_indexes = matrix_t.columns
    db_indexes = matrix_p.columns

    # 创建模型
    model = gurobipy.Model('two_level_assignment')
    matrix_X = model.addVars(job_indexes, resource_indexes, vtype=gurobipy.GRB.BINARY, name="matrix_X")
    matrix_Y = model.addVars(job_indexes, db_indexes, vtype=gurobipy.GRB.BINARY, name="matrix_Y")
    model.update()

    # 设置目标函数
    model.setObjective(
        # sum(matrix_X[i, j] * (1 / matrix_t.at[i, j]) for j in resource_indexes) * matrix_Y[i, k] for i in job_indexes,
        gurobipy.quicksum((matrix_X[i, j] / matrix_t.at[i, j]) * matrix_Y[i, k] for i in job_indexes for j in resource_indexes for k in db_indexes),
        sense=gurobipy.GRB.MAXIMIZE
    )

    # 创建约束条件
    model.addConstrs(gurobipy.quicksum(matrix_Y[i, k] for k in db_indexes) <= 1 for i in job_indexes)
    model.addConstrs(gurobipy.quicksum(matrix_X[i, j] for j in resource_indexes) <= 1 for i in job_indexes)

    # 执行最优化
    model.optimize()

    result_X = matrix_t * 0
    result_Y = matrix_p * 0
    if model.status == gurobipy.GRB.Status.OPTIMAL:
        solution_X = [k for k, v in model.getAttr('matrix_X', matrix_X).items() if v == 1]
        solution_Y = [k for k, v in model.getAttr('matrix_Y', matrix_Y).items() if v == 1]
        for i, j in solution_X:
            print(f"{i} -> {j}: {cost_matrix.at[i,j]}")
            result_X.at[i, j] = 1
        for i, k in solution_Y:
            print(f"{i} -> {k}: {cost_matrix.at[i,k]}")
            result_Y.at[i, k] = 1
	# return result

    # print("status: ", model.status)
    print("result_X: ", result_X)
    print("result_Y: ", result_Y)

def cvxpy_sol_ILP(NUM_JOB, NUM_RESOURCE, NUM_DB,
                matrix_t, matrix_p, ones_t, ones_p, alpha,
                solver = 'ECOP'):
    matrix_X = cp.Variable((NUM_JOB, NUM_RESOURCE))
    matrix_Y = cp.Variable((NUM_JOB, NUM_DB))
    

    objective = cp.Maximize(
        # cp.sum(cp.multiply(cp.sum(matrix_Y, axis=1), cp.sum(cp.multiply(matrix_t, matrix_X), axis=1)))
        # cp.sum(alpha * cp.sum(cp.multiply(matrix_p, matrix_Y), axis=1) + (1 - alpha) * cp.sum(cp.multiply(matrix_t, matrix_X), axis=1))
        cp.sum(cp.multiply(matrix_X @ ones_t, matrix_Y @ ones_p))
    )
    constraints = [
        matrix_X >= 0,
        matrix_Y >= 0,
        cp.sum(matrix_X, axis=1) <= 1,
        cp.sum(matrix_Y, axis=1) <= 1,
    ]
    # constraints = []

    cvxprob = cp.Problem(objective, constraints)
    result = cvxprob.solve(solver)
    print(matrix_X.value)
    print(matrix_Y.value)

    if cvxprob.status != "optimal":
        print('WARNING: Allocation returned by policy not optimal!')

if __name__ == "__main__":
    NUM_JOB = 20
    NUM_RESOURCE = 3
    NUM_DB = 2

    matrix_t = np.random.randint(1, 5, size=(NUM_JOB, NUM_RESOURCE))
    matrix_p = np.random.randint(1, 5, size=(NUM_JOB, NUM_DB))
    ones_t = np.ones(shape=(NUM_RESOURCE, 1))
    ones_p = np.ones(shape=(NUM_DB, 1))

    job_indexes = ['J{}'.format(i+1) for i in range(NUM_JOB)]
    resource_indexes = ['R{}'.format(i+1) for i in range(NUM_RESOURCE)]
    db_indexes = ['D{}'.format(i+1) for i in range(NUM_DB)]
    df_matrix_t = pd.DataFrame(matrix_t, index=job_indexes, columns=resource_indexes)
    df_matrix_p = pd.DataFrame(matrix_p, index=job_indexes, columns=db_indexes)

    alpha = 0.5

    gurobipy_sol_ILP(NUM_JOB, NUM_RESOURCE, NUM_DB,
                df_matrix_t, df_matrix_p, alpha)