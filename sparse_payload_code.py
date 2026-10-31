"""Experimental systematic repeat-accumulate code; NumPy only, no image claims."""
import numpy as np


class SparsePayloadCode:
    # Eight blocks use 8192 information bits and 6208 parity bits in total.
    k = 1024
    m = 776
    n = k + m

    def __init__(self, seed=20261031):
        rng = np.random.default_rng(seed)
        self.a_checks = np.array([rng.choice(self.m, 3, replace=False) for _ in range(self.k)])
        self.a_vars = np.repeat(np.arange(self.k), 3)
        self.a_rows = self.a_checks.ravel()
        checks = [[] for _ in range(self.m)]
        for row, var in zip(self.a_rows, self.a_vars):
            checks[row].append(int(var))
        for row in range(self.m):
            checks[row].append(self.k+row)
            if row:
                checks[row].append(self.k+row-1)
        self.variables = np.zeros((self.m,max(map(len,checks))),dtype=int)
        self.valid = np.zeros_like(self.variables,dtype=bool)
        for row, vs in enumerate(checks):
            self.variables[row,:len(vs)] = vs
            self.valid[row,:len(vs)] = True

    def _binary(self, bits, width):
        bits = np.asarray(bits)
        if bits.ndim != 2 or bits.shape[1] != width or not np.all((bits==0)|(bits==1)):
            raise ValueError('Expected a binary batch with width '+str(width))
        return bits.astype(np.uint8)

    def encode(self, message):
        message = self._binary(message,self.k)
        rhs = np.zeros((len(message),self.m),dtype=np.uint8)
        for i in range(len(message)):
            np.bitwise_xor.at(rhs[i],self.a_rows,message[i,self.a_vars])
        parity = np.bitwise_xor.accumulate(rhs,axis=1)
        return np.concatenate((message,parity),axis=1)

    def syndrome(self, codeword):
        codeword = self._binary(codeword,self.n)
        return np.sum(codeword[:,self.variables]*self.valid,axis=2)%2

    def decode(self, llr, max_iterations=60, normalization=.8):
        # Positive LLR means bit zero. The decoder never receives a reference message.
        llr = np.asarray(llr,dtype=float)
        if llr.ndim!=2 or llr.shape[1]!=self.n or not np.isfinite(llr).all():
            raise ValueError('Expected finite batch LLRs')
        if type(max_iterations) is not int or max_iterations<0 or not 0<normalization<=1:
            raise ValueError('Invalid decoding settings')
        channel = np.clip(llr,-30,30)
        hard = (channel<0).astype(np.uint8)
        answers = hard[:,:self.k].copy()
        converged = ~self.syndrome(hard).any(axis=1)
        iterations = np.zeros(len(llr),dtype=int)
        q = channel[:,self.variables].copy()
        q[:,~self.valid] = 0
        edge_variables = self.variables[self.valid]
        batch = np.arange(len(llr))[:,None]
        for iteration in range(1,max_iterations+1):
            if converged.all():
                break
            magnitudes = np.where(self.valid,np.abs(q),np.inf)
            minimum_index = magnitudes.argmin(axis=2)
            minimum = magnitudes.min(axis=2)
            np.put_along_axis(magnitudes,minimum_index[:,:,None],np.inf,axis=2)
            second = magnitudes.min(axis=2)
            outgoing = np.broadcast_to(minimum[:,:,None],q.shape).copy()
            np.put_along_axis(outgoing,minimum_index[:,:,None],second[:,:,None],axis=2)
            signs = np.where(q<0,-1.,1.)
            r = normalization*outgoing*signs*np.prod(signs,axis=2)[:,:,None]
            r[:,~self.valid] = 0
            posterior = channel.copy()
            np.add.at(posterior,(batch,edge_variables[None,:]),r[:,self.valid])
            hard = (posterior<0).astype(np.uint8)
            valid_code = ~self.syndrome(hard).any(axis=1)
            accepted = valid_code & ~converged
            answers[accepted] = hard[accepted,:self.k]
            iterations[accepted] = iteration
            converged |= valid_code
            q = np.clip(posterior[:,self.variables]-r,-30,30)
            q[:,~self.valid] = 0
        # Explicit non-convergence policy: return raw systematic decisions, not oracle correction.
        iterations[~converged] = max_iterations
        return answers,dict(converged=converged,iterations=iterations,
                            raw_fallback=~converged)
