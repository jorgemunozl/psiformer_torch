from flax import nnx
import optax
import jax
import jax.numpy as jnp

# Function to learn
def function_to_learn(x: jnp.ndarray) -> jnp.ndarray:
    return x**2

# random key

class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dmid, rngs=rngs)
        # self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        #self.dropout = nnx.Dropout(0.5)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, rngs: nnx.Rngs) -> jnp.ndarray:
        x = self.linear(x)
        # x = self.bn(x)
        # x = nnx.relu(x)
        x = jax.nn.sigmoid(x)
        #x = self.dropout(x, rngs=rngs)
        x = self.linear_out(x)
        return x


# Initialize Rngs with params stream for model initialization
init_rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
model = Model(din=1, dmid=10, dout=1, rngs=init_rngs)
output = model(jnp.array([100.0]), init_rngs)
print(output)

"""
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001), wrt=nnx.Param)

def train_step(model, optimizer, x: jnp.ndarray, y: jnp.ndarray, dropout_key: jax.Array):
    def loss_fn(model):
        # Create Rngs inside the trace context to avoid mutation errors
        rngs = nnx.Rngs(dropout=dropout_key)
        pred = model(x, rngs)
        return jnp.mean((pred - y) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Generate training data
x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
y_data = function_to_learn(x_data)

# Training loop - split keys and pass to train_step
rng_key = jax.random.PRNGKey(1)
for epoch in range(100):
    # Split key for each step
    rng_key, dropout_key = jax.random.split(rng_key)
    loss = train_step(model, optimizer, x_data, y_data, dropout_key)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test prediction
test_x = jnp.array([[2.5], [3.5]])
test_rng_key, _ = jax.random.split(rng_key)
test_rngs = nnx.Rngs(dropout=test_rng_key)
test_y = model(test_x, test_rngs)
print(f"\nPredictions for x={test_x.flatten()}: {test_y.flatten()}")
print(f"True values: {function_to_learn(test_x).flatten()}")
"""