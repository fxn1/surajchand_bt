import math
from statistics import NormalDist

# -----------------------------
# Black–Scholes call price + delta
# -----------------------------

class BlackScholesModel:
    def __init__(self, q: float = 0.0):
        """
        Initialize the Black-Scholes model with optional dividend yield.
        :param q: Dividend yield (default is 0.0).
        """
        self.q = q
        self.N = NormalDist()

    def bs_call_price_delta(self, S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
        """
        Calculate the Black-Scholes price and delta for a European call option.
        :param S: Current stock price.
        :param K: Strike price.
        :param T: Time to maturity (in years).
        :param r: Risk-free rate.
        :param sigma: Volatility.
        :return: Tuple of (call price, delta).
        """
        if T <= 0 or sigma <= 0 or not math.isfinite(sigma):  ## expired option has intrinsic value only
            call = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
            return call, delta

        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1, Nd2 = self.N.cdf(d1), self.N.cdf(d2)

        call = S * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
        delta = math.exp(-q * T) * Nd1
        return call, delta


    def strike_for_delta_call(self, S: float, target_delta: float, T: float, r: float, q: float, sigma: float,
                              strike_round: float = 1.0) -> float:
        """
        Solve strike K that gives target call delta under BS:
        delta = exp(-qT) * N(d1)
            :param S: Current stock price.
            :param target_delta: Desired delta.
            :param T: Time to maturity (in years).
            :param r: Risk-free rate.
            :param sigma: Volatility.
            :param strike_round: Rounding increment for the strike price.
            :return: Strike price.
        """
        adj = target_delta * math.exp(q * T)
        adj = min(max(adj, 1e-6), 1 - 1e-6)
        d1 = self.N.inv_cdf(adj)

        ln_S_over_K = d1 * sigma * math.sqrt(T) - (r - q + 0.5 * sigma * sigma) * T
        K = S / math.exp(ln_S_over_K)

        if strike_round and strike_round > 0:
            K = round(K / strike_round) * strike_round
            K = max(strike_round, K)

        return float(K)
