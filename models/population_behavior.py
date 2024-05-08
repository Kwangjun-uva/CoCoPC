import numpy as np
import matplotlib.pyplot as plt
from parameters import dt
from practice_scalar import dr


if __name__ == "__main__":

    T = 1.0
    T_steps = int(T / dt)

    # test pPE and nPE

    r_ppe_e = np.zeros(T_steps)
    r_ppe_i = np.zeros(T_steps)
    bu_input = np.zeros(T_steps)
    input_vals = np.arange(0, 30.1, 0.1)

    basal_re = []
    avg_re = []
    basal_ri = []
    avg_ri = []
    for j, inp_val in enumerate(input_vals):
        bu_input[300:800] = inp_val
        for i in range(T_steps - 1):
            r_ppe_e[i + 1] = dr(r=r_ppe_e[i], inputs=bu_input[i], ei_type='exc')
            r_ppe_i[i + 1] = dr(r=r_ppe_i[i], inputs=bu_input[i])
        basal_re.append(r_ppe_e[:300].mean())
        avg_re.append(r_ppe_e[300:800].mean())
        basal_ri.append(r_ppe_i[:300].mean())
        avg_ri.append(r_ppe_i[300:800].mean())
    plt.plot(input_vals, avg_re, c='b', label='exc')
    plt.plot(input_vals, avg_ri, c='r', label='inh')
    plt.xlabel('input')
    plt.ylabel('output (fr: Hz)')
    plt.legend()
    plt.show()

    plt.plot(input_vals, basal_re, c='b', label='exc')
    plt.plot(input_vals, basal_ri, c='r', label='inh')
    plt.xlabel('input')
    plt.ylabel('basal (fr: Hz)')
    plt.legend()
    plt.show()