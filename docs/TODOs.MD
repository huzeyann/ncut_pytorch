
### pytorch code package
    [ ] ncut
        [ ] inverse transform
    [^] predictors
        [x] vision predictor  `ncut_pytorch/predictor/vision_predictor.py`
        [ ] language predictor
        [ ] vision-language predictor
    [ ] super-resolution models
        [x] dinov1, dinov2, dinov2reg
        [ ] open_clip
    [ ] feature align

### notebooks
    [x] preview hierarchy
    [x] predict segments
    [x] mspace color hierarchy
    [x] custom model

### homepage
    [ ] a "k" slider
    [ ] hierarchy preciew clicker
    [ ] feature align (diffusion)

### Methods documentation
    [ ] ncut method
        [ ] basic ncut idea
            [ ] the sample balacing issue in ncut
        [ ] old nystrom ncut
            [ ] when does the old nystrom fails: 1. sample unbalance, 2. compelexity scaling
        [ ] new nystrom ncut 
            [ ] how it solve 1. sample balancing, 2. complexity scaling
    [ ] kway ncut
    [ ] mspace method
    [ ] feature align method

### Tutorials
    [ ] tutorial 1: feature visualization
        [ ] extract DINOv3 features
        [ ] ncut eigvec, show in raw, make a 11x6 plot, 6 images, 10 eigvec
        [ ] tsne coloring, reduce 10 eigvec into 3D
        [ ] mspace coloring reduce 10 eigvec into 3D
        [ ] convinence NcutVisionPredictor class
    [^] tutorial 2: discrete ncut
        [x] extract DINOv3 features
        [x] ncut eigvec, show in raw, make a 11x6 plot, 6 images, 10 eigvec
        [x] kway ncut eigvec, show in raw, make 11x6 plot, 6 images, 10 eigvec
        [x] show cluster mask, cluster centroids
        [-] discretize the mspace RGB, use mean RGB within each mask to fill the whole
        [-] convinence NcutVisionPredictor class
    [ ] tutorial 3: hierarchy in ncut
        [ ] kway ncut at different k
        [ ] click on a point, show mask at different k
        [ ] convinence NcutVisionPredictor class
    [ ] tutorial 4: feature align
        [ ] mutiple models: dino, clip
        [ ] diffusion layer x timesteps

### Convinece NcutPredictors
    [ ] vision
    [ ] language
    [ ] vision-language

### demo
    [ ] hierarchy preview
    [ ] coloring loading animation

### other code package?
    [ ] matlab
    [ ] R