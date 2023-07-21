import numpy as np 
import pandas as pd

def getName():
    return "Mehmet Akif Baba"

def getStudentID():
    return "070200743"
dic = {"value": [79,32,47,18,26,85,33,40,45,59],    
       "weights": [85,26,48,21,22,95,43,45,55,52]}

df = pd.DataFrame(dic)
df_data=df
max_weight = 300
def generate_initial_solution(df_data, num_elements, max_weight):
    current_sol=np.zeros(num_elements)
    num_selected = int(np.floor(num_elements/3))
    selected_indices = np.random.choice(num_elements, num_selected, replace=False)
    current_sol[selected_indices] = 1  
    return current_sol
def generate_new_sol(current_sol,num_elements):
    random_index = np.random.randint(num_elements)
    solution = current_sol.copy()
    solution[random_index] = 1 - solution[random_index]
    return solution
def calculate_objective(df_data, solution, max_weight):
    total_weight = np.sum(solution * df_data['weights'])
    if total_weight > max_weight:
        return 0
    else:
        total_value = np.sum(solution * df_data['value'])
        return total_value
def knapsack_simulated_annealing(df_data, max_weight, num_iter = 100, best_obj=False, initial_temp = 10, cooling_rate = 0.99, random_seed = 42):
    np.random.seed(random_seed)
    num_elements=len(df_data)
    current_sol = generate_initial_solution(df_data, num_elements, max_weight)
    current_obj = calculate_objective(df_data, current_sol, max_weight)
    best_sol = current_sol
    best_obj_val = current_obj
    for i in range(num_iter):        
        new_sol = generate_new_sol(current_sol, num_elements)
        new_obj_val = calculate_objective(df_data,new_sol, max_weight)
        diff = new_obj_val - current_obj
        initial_temp = initial_temp * cooling_rate
        accept_prob = np.exp(diff/initial_temp)   
        if diff > 0:
            current_sol = new_sol
            current_obj = new_obj_val
            best_obj_val = current_obj
            best_sol = current_sol
        else:
            if np.random.uniform(0,1,size=1)<accept_prob:
                current_sol=new_sol
                current_obj=new_obj_val
                best_obj_val = current_obj
                best_sol = current_sol
    if best_obj:
        return best_sol.astype(int), int(best_obj_val)
    else:
        return best_sol.astype(int)
sonuc=np.zeros(len(df_data))       
sonuc = knapsack_simulated_annealing(df_data, max_weight)

