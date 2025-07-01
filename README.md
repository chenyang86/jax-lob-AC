# JAX-LOB-August Chen version

## JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading

### Architecture of the package:
* `gymnax_exchange`: the `GPU` version of rl_environment
  * `jaxob`: Jax limit order book
  * `jaxen`: Jax trading_environment
  * `jaxrl`: Jax training loop

## Dependencies

```
!pip install gymnax #0.0.9
!pip install distrax brax
!pip install rlax

```


