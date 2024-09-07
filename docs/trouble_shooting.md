


### Error installing from `pip`

`fpsample` is a dependency for `ncut-pytorch`, if you running into the issue like this image, please try the following steps.

<div style="text-align: center;">
<img src="../images/rust_error_fpsample.png" style="width:100%;">
</div>

Step1: (**optional**) make sure CC compiler is installed by installing `build-essential`. you can skip this step if `cc --version` and `g++ --version` are working on your machine.

```shell
sudo apt-get update
sudo apt-get install build-essential
```


Step2: install `rustc` ([https://rustup.rs/](https://rustup.rs/)), use command:
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# press enter to install
```
```shell
. "$HOME/.cargo/env"
# activate `rustc`
```

Step3: re-try installing `ncut-pytorch` with `fpsample`
```
pip install ncut-pytorch -U
```

Finally, if you still run into other errors when installing `fpsample`, please follow their instruction and build from source: [https://github.com/leonardodalinky/fpsample](https://github.com/leonardodalinky/fpsample)