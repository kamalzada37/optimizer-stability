# run_grid_light.ps1

$optimizers = @('sgd','adam')
$noises     = @(0.0, 0.1, 0.3)
$precisions = @('float32','float64')
$seeds      = @(0,1)

foreach ($opt in $optimizers) {
    foreach ($noise in $noises) {
        foreach ($prec in $precisions) {
            foreach ($seed in $seeds) {
                $lr = if ($opt -eq 'adam') { 0.001 } else { 0.01 }
                Write-Output "Run: $opt noise=$noise prec=$prec seed=$seed"
                .\.venv\Scripts\python.exe -m src.train `
                    --optimizer $opt `
                    --lr $lr `
                    --noise $noise `
                    --precision $prec `
                    --seed $seed `
                    --epochs 3 `
                    --outdir .\results\light `
                    --dataset mnist
            }
        }
    }
}
