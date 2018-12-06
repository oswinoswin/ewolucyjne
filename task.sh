#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J Ewolution
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=12
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=4GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plguchwat1112018a
## Specyfikacja partycji
#SBATCH -p plgrid-testing
## Plik ze standardowym wyjściem
#SBATCH --output="ewolution_output.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="ewolution_error.err"


## Zaladowanie modulu IntelMPI w wersji domyslnej
module add plgrid/tools/python-intel/3.6.5

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

pip install --user deap

echo "islands_count,time,best_fitness" > ewolution_results.csv

for rep in {1..5}
do
    for islands in {5..50..5}
    do
        python main.py $islands >> ewolution_results.csv
    done
done

