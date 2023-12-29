use anyhow::Result;
use dfdx::data::OneHotEncode;
use dfdx::prelude::*;
use tokenizers::Tokenizer;

const VOCAB_SIZE: usize = 32000;
const HIDDEN_SIZE: usize = 64;
const DECODER_BLOCKS: usize = 1;

fn generate_positional_embeddings<D: Device<f32>>(
    batch_size: usize,
    len: usize,
    dev: &D,
) -> Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D> {
    // We can only index on the CPU
    // Shoutout to https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    let cpu_dev: Cpu = Default::default();
    let mut positional_embeddings: Tensor<(usize, Const<64>), f32, Cpu> =
        cpu_dev.ones_like(&(len, Const));
    for k in 0..len {
        for i in 0..HIDDEN_SIZE / 2 {
            let denominator = 10000.0_f32.powf(2. * (i as f32 / (HIDDEN_SIZE as f32)));
            positional_embeddings[[k, 2 * i]] = (k as f32 / denominator).sin();
            positional_embeddings[[k, 2 * i + 1]] = (k as f32 / denominator).cos();
        }
    }
    let positional_embeddings = positional_embeddings.to_device(dev);
    (0..batch_size)
        .map(|_i| positional_embeddings.clone())
        .collect::<Vec<Tensor<(usize, Const<HIDDEN_SIZE>), f32, D>>>()
        .stack()
}

struct DecoderBlock<D: Device<f32>> {
    dev: D,
    qkv_weights: Tensor<Rank2<HIDDEN_SIZE, { HIDDEN_SIZE * 3 }>, f32, D>,
    linear_weights: Tensor<Rank2<HIDDEN_SIZE, HIDDEN_SIZE>, f32, D>,
}

impl<D: Device<f32>> DecoderBlock<D> {
    fn new(dev: D) -> Self {
        // This is not how we should be doing initializing the weights lol
        let qkv_weights = dev.sample_normal() / 10.;
        let linear_weights = dev.sample_normal() / 10.;
        Self {
            dev,
            qkv_weights,
            linear_weights,
        }
    }

    fn forward<T: Tape<f32, D> + std::fmt::Debug>(
        &mut self,
        input: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T>,
    ) -> Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T> {
        let qkv_proj = input.matmul(self.qkv_weights.clone());
        let q_proj: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T> = qkv_proj
            .retaped()
            .slice((0.., 0.., 0..HIDDEN_SIZE))
            .realize();
        let k_proj: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T> = qkv_proj
            .retaped()
            .slice((0.., 0.., HIDDEN_SIZE..HIDDEN_SIZE * 2))
            .realize();
        let k_proj: Tensor<(usize, Const<HIDDEN_SIZE>, usize), f32, D, T> =
            k_proj.permute::<_, Axes3<0, 2, 1>>();
        let v_proj: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T> = qkv_proj
            .slice((0.., 0.., HIDDEN_SIZE * 2..HIDDEN_SIZE * 3))
            .realize();
        let scores: Tensor<(usize, usize, usize), f32, D, T> = q_proj.matmul(k_proj);
        let mask: Tensor<(usize, usize, usize), f32, D> =
            self.dev.upper_tri_like(&scores, f32::MIN, 1);
        let masked_scores: Tensor<(usize, usize, usize), f32, D, T> = scores + mask;
        let softmaxed_scores: Tensor<(usize, usize, usize), f32, D, T> =
            masked_scores.softmax::<Axis<2>>();
        let f: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, T> =
            softmaxed_scores.matmul(v_proj);
        f.matmul(self.linear_weights.clone())
    }

    fn update(&mut self, gradients: &Gradients<f32, D>) {
        // Update the qkv weights
        let qkv_gradients = gradients.get(&self.qkv_weights);
        self.qkv_weights = self.qkv_weights.clone() - (qkv_gradients * 0.1);

        // Update the linear weights
        let linear_gradients = gradients.get(&self.linear_weights);
        self.linear_weights = self.linear_weights.clone() - (linear_gradients * 0.1);
    }
}

struct Model<D: Device<f32>> {
    dev: D,
    embed_weights: Tensor<Rank2<VOCAB_SIZE, HIDDEN_SIZE>, f32, D>,
    decoder_blocks: Vec<DecoderBlock<D>>,
    projection_weights: Tensor<Rank2<HIDDEN_SIZE, VOCAB_SIZE>, f32, D>,
}

impl<D: Device<f32>> Model<D> {
    fn new(dev: D) -> Self {
        // Once again this is not how we should be initializing the weights
        let embed_weights = dev.sample_uniform() / 10.;
        let projection_weights = dev.sample_uniform() / 10.;
        let decoder_blocks = (0..DECODER_BLOCKS)
            .map(|_| DecoderBlock::new(dev.clone()))
            .collect();
        Self {
            embed_weights,
            decoder_blocks,
            projection_weights,
            dev,
        }
    }

    fn forward(&mut self, token_ids: Vec<Vec<usize>>) -> Result<Vec<usize>> {
        let positional_embeddings =
            generate_positional_embeddings(token_ids.len(), token_ids[0].len(), &self.dev);

        let embedding_inputs: Vec<Tensor<(usize, Const<32000>), f32, _>> = token_ids
            .iter()
            .map(|t| self.dev.one_hot_encode(Const::<32000>, t.clone()))
            .collect();
        let embedding_inputs: Tensor<(usize, usize, Const<32000>), f32, _> =
            embedding_inputs.stack();

        let token_embeddings: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D> =
            embedding_inputs.matmul(self.embed_weights.clone());

        // Add them together to get the final embeddings matrix
        let mut x: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D> =
            token_embeddings + positional_embeddings;

        // Iterate through our decode layers
        for layer in self.decoder_blocks.iter_mut() {
            x = x.retaped() + layer.forward(x);
        }

        // Get the final output
        let logits = x.matmul(self.projection_weights.clone());

        // Get the predicted tokens
        // We don't care about the logits for the other word predictions, just the last one
        // While training, we do care about the logits for other word predictions
        let mut predicted_tokens = vec![];
        for i in 0..token_ids.len() {
            let predicted_token = logits
                .clone()
                .slice((i..i + 1, token_ids[0].len() - 1.., 0..))
                .as_vec()
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |(best_index, best_value), (index, value)| {
                    if *value > best_value {
                        (index, *value)
                    } else {
                        (best_index, best_value)
                    }
                })
                .0;
            predicted_tokens.push(predicted_token);
        }
        Ok(predicted_tokens)
    }

    fn train(&mut self, token_ids: Vec<Vec<usize>>, labels: Vec<Vec<usize>>) -> Result<f32> {
        let grads = Gradients::leaky();

        let positional_embeddings =
            generate_positional_embeddings(token_ids.len(), token_ids[0].len(), &self.dev);

        // Generate the token embeddings
        let embedding_inputs: Vec<Tensor<(usize, Const<32000>), f32, _>> = token_ids
            .iter()
            .map(|t| self.dev.one_hot_encode(Const::<32000>, t.clone()))
            .collect();
        let embedding_inputs: Tensor<(usize, usize, Const<32000>), f32, _> =
            embedding_inputs.stack();

        let token_embeddings: Tensor<
            (usize, usize, Const<HIDDEN_SIZE>),
            f32,
            D,
            OwnedTape<f32, D>,
        > = embedding_inputs
            .trace(grads)
            .matmul(self.embed_weights.clone());

        // Add them together to get the final embeddings matrix
        let mut x: Tensor<(usize, usize, Const<HIDDEN_SIZE>), f32, D, OwnedTape<f32, D>> =
            token_embeddings + positional_embeddings;

        // Iterate through our decode layers
        for layer in self.decoder_blocks.iter_mut() {
            x = x.retaped() + layer.forward(x);
        }

        // Calculate the loss
        let logits = x.matmul(self.projection_weights.clone());
        let labels: Vec<Tensor<(usize, Const<32000>), f32, _>> = labels
            .into_iter()
            .map(|l| self.dev.one_hot_encode(Const::<32000>, l))
            .collect();
        let labels: Tensor<(usize, usize, Const<32000>), f32, _> = labels.stack();
        let loss = cross_entropy_with_logits_loss(logits, labels);

        // Store the loss before going backwards so we can return it
        let real_loss = loss.as_vec()[0];

        // Get the gradients
        let gradients = loss.backward();

        // Update the embedding weights
        let embed_gradients = gradients.get(&self.embed_weights);
        self.embed_weights = self.embed_weights.clone() - (embed_gradients * 0.1);

        // Update the projection weights
        let projection_gradients = gradients.get(&self.projection_weights);
        self.projection_weights = self.projection_weights.clone() - (projection_gradients * 0.1);

        // Update the weights in all of the decoder blocks
        for layer in self.decoder_blocks.iter_mut() {
            layer.update(&gradients);
        }

        Ok(real_loss)
    }
}

fn main() -> Result<()> {
    let tokenizer: Tokenizer =
        Tokenizer::from_file("./tokenizer.json").map_err(anyhow::Error::msg)?;
    println!(
        "Loaded Tokenizer - Tokenizer vocab size {:?}",
        tokenizer.get_vocab_size(true)
    );

    let tokens = tokenizer
        .encode("AI is going to", false)
        .map_err(anyhow::Error::msg)?;
    let token_ids: Vec<usize> = tokens.get_ids().iter().map(|t| *t as usize).collect();

    let dev: Cuda = Default::default();
    let mut model = Model::new(dev);

    let labels = vec![vec![29902, 338, 2675, 304, 1735]];
    for i in 0..2500 {
        let loss = model.train(vec![token_ids.clone()], labels.clone())?;
        println!("{i} - {loss}");
    }

    let output = model.forward(vec![vec![319]])?;
    println!("{:?}", output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx::tensor_ops::Backward;

    // Used to test against pytorch
    #[test]
    fn it_works() -> anyhow::Result<()> {
        let dev: Cpu = Default::default();

        // To keep things simple we will keep gradients for all temporary tensors
        let grads = Gradients::leaky();

        // We have to add grads to the inputs or can't matmul with them
        let inputs: Tensor<Rank2<2, 4>, f32, _, _> =
            dev.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]]);

        let t1: Tensor<Rank2<4, 4>, f32, _, _> = dev.tensor([
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8],
            [1.9, 2.0, 2.1, 2.2],
            [2.3, 2.4, 2.5, 2.6],
        ]);

        let t2: Tensor<Rank2<2, 2>, f32, _, _> = dev.tensor([[1.1, 1.2], [1.3, 1.4]]);
        let t3: Tensor<Rank2<2, 2>, f32, _, _> = dev.tensor([[1.5, 1.6], [1.7, 1.8]]);

        let t4: Tensor<Rank2<2, 4>, f32, _, OwnedTape<f32, _>> =
            inputs.trace(grads).matmul(t1.clone());

        let t6: Tensor<Rank2<2, 2>, f32, _, OwnedTape<f32, _>> =
            t4.retaped().slice((0.., 0..2)).realize();
        let t7: Tensor<Rank2<2, 2>, f32, _, _> = t4.slice((0.., 2..4)).realize();

        let t8 = t2.trace(Gradients::leaky()).matmul(t6);
        let t9 = t3.trace(Gradients::leaky()).matmul(t7);
        let t10 = t9.matmul(t8);

        let res = t10.mean();

        println!("MEAN: {:?}", res.as_vec());

        let mut grads: Gradients<f32, Cpu> = res.backward();

        let inputs_gradients = grads.get_or_alloc_mut(&inputs)?;
        println!("inputs - {:?}", inputs_gradients.as_slice());

        let t1_gradients = grads.get_or_alloc_mut(&t1)?;
        println!("t1 - {:?}", t1_gradients.as_slice());
        let t2_gradients = grads.get_or_alloc_mut(&t2)?;
        println!("t2 - {:?}", t2_gradients.as_slice());
        let t3_gradients = grads.get_or_alloc_mut(&t3)?;
        println!("t3 - {:?}", t3_gradients.as_slice());

        // let qk_gradients = grads.get_or_alloc_mut(&qk)?;
        // println!("qk - {:?}", qk_gradients.as_slice());

        // let result = (t1.trace(t1_grads)) * (t2.trace(t2_grads));

        // let mut grads = result.backward();

        // let t1_gradients = grads.get_or_alloc_mut(&t1)?;
        // println!("T1 - {:?}", t1_gradients.as_slice());

        // let t2_gradients = grads.get_or_alloc_mut(&t2)?;
        // println!("T2 - {:?}", t2_gradients.as_slice());

        Ok(())
    }
}
