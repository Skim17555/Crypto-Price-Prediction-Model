





def SIR_model(n, beta, gamma):

    S,I,R = n

    #ds_dt = -beta*S*I


    di_dt = 


    #dr_dt = gamma * I

    return di_dt


#average total score: 2586
S0 = 2243

I0 = 210 

R0 = 133
beta = 180    #doller per sentiment
gamma = 50

print(SIR_model([S0, I0, R0], beta, gamma))