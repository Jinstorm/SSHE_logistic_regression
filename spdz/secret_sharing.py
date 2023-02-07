import random
import numpy as np

def getAdditiveShares(secret, N, fieldSize):
	'''Generate N additive shares from 'secret' in finite field of size 'fieldSize'.'''

	# Generate n-1 shares randomly
	shares = [random.randrange(fieldSize) for i in range(N-1)]

	# Append final share by subtracting all shares from secret
	# Modulo is done with fieldSize to ensure share is within finite field
	shares.append((secret - sum(shares)) % fieldSize )
	return shares

def reconstructSecret(shares, fieldSize):
	'''Regenerate secret from additive shares'''
	return sum(shares) % fieldSize

if __name__ == "__main__":
	# Generating the shares
	share = getAdditiveShares(4321, 2, 10**6)
	shares = getAdditiveShares(1234, 2, 10**6)
	# print('Shares are:', shares, share)
	# result = [1,2] #shares + share
	# result[0] = share[0]*shares[0]
	# result[1] = share[1]*shares[1]

	share = share*2
	shares = shares*0
	result = share+shares
	# Reconstructing the secret from shares
	print('Reconstructed secret:', reconstructSecret(result, 10**5))