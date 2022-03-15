import numpy as np
import numba as nb

@nb.njit
def ClassShift(class_arr):
    shape_tuple = np.shape(class_arr)
    SCtSD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SCtUC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SCtUD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SDtSC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SDtUC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SDtUD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UCtSC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UCtSD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UCtUD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UDtSC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UDtSD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UDtUC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SCtSC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    SDtSD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UCtUC = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    UDtUD = np.zeros((shape_tuple[0], shape_tuple[1], shape_tuple[2]))
    hit = 0.
    miss = 0.
    for var in range(shape_tuple[0]):
        print(var)
        for it in range(shape_tuple[1]):
            for i_main in range(shape_tuple[2]-1):

                classd = np.divide(class_arr[var, it, i_main, :],
                                   class_arr[var, it, i_main+1, :])
                classm = np.multiply(class_arr[var, it, i_main, :],
                                     class_arr[var, it, i_main+1, :])
                classa = np.add(class_arr[var, it, i_main, :],
                                class_arr[var, it, i_main+1, :])
                classs = np.subtract(class_arr[var, it, i_main, :],
                                     class_arr[var, it, i_main+1, :])
                
                classA = np.multiply(classd, classa)
                classB = np.multiply(classm, classs)
                
                for npl in range(shape_tuple[3]):    

                    #SC to others
                    if classA[npl][0] == -4. and classB[npl][0] == -48.:
                        SCtSD[var, it,
                              i_main+1] = SCtSD[var, it,
                                                i_main+1] + 1.
                        hit = hit + 1
                    elif classA[npl][0] == 12. and classB[npl][0] == 16:
                        SCtUC[var, it,
                              i_main+1] = SCtUC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 0 and classB[npl][0] == -128:
                        SCtUD[var, it,
                              i_main+1] = SCtUD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1

                    #SD to others
                    elif classA[npl][0] == -1 and classB[npl][0] == 48:
                        SDtSC[var, it,
                              i_main+1] = SDtSC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 0 and classB[npl][0] == 16:
                        SDtUC[var, it,
                              i_main+1] = SDtUC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == -3 and classB[npl][0] == 16:
                        SDtUD[var, it,
                              i_main+1] = SDtUD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1

                    #UC to others
                    elif classA[npl][0] == 3 and classB[npl][0] == -16:
                        UCtSC[var, it,
                              i_main+1] = UCtSC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 0 and classB[npl][0] == -16:
                        UCtSD[var, it,
                              i_main+1] = UCtSD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 1 and classB[npl][0] == -48:
                        UCtUD[var, it,
                              i_main+1] = UCtUD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1

                    # UD to others
                    elif classA[npl][0] == 0 and classB[npl][0] == 128:
                        UDtSC[var, it,
                              i_main+1] = UDtSC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == -12 and classB[npl][0] == -16:
                        UDtSD[var, it,
                              i_main+1] = UDtSD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 4 and classB[npl][0] == 48:
                        UDtUC[var, it,
                              i_main+1] = UDtUC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                        
                    #Self loops
                    elif classA[npl][0] == 8 and classB[npl][0] == 0:
                        SCtSC[var, it,
                              i_main+1] = SCtSC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == -4 and classB[npl][0] == 0:
                        SDtSD[var, it,
                              i_main+1] = SDtSD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == 4 and classB[npl][0] == 0:
                        UCtUC[var, it,
                              i_main+1] = UCtUC[var, it,
                                                i_main+1] +1.
                        hit = hit + 1
                    elif classA[npl][0] == -8 and classB[npl][0] == 0:
                        UDtUD[var, it,
                              i_main+1] = UDtUD[var, it,
                                                i_main+1] +1.
                        hit = hit + 1

                    else:
                        #print(classA[npl][0],classB[npl][0])
                        miss = miss + 1 # Problematic when s=0.
    print(miss/(miss+hit), miss)
    return(SCtSD, SCtUC, SCtUD, SDtSC, SDtUC, SDtUD,
           UCtSC, UCtSD, UCtUD, UDtSC, UDtSD, UDtUC,
           SCtSC, SDtSD, UCtUC, UDtUD)
