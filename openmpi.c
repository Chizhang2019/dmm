#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_LINES 5000 /* the max number of elements in product matrix*/
#define MASTER 0			 /* taskid of first task */
#define FROM_MASTER 1  /* setting a message type */
#define FROM_WORKER 2  /* setting a message type */

int main (int argc, char *argv[]){
	int numtasks, taskid,	numworkers,	source, dest, mtype, rc;
	int alines, blines;
	int *ay, *ax, *by, *bx; /* cols, rows in MatA and MatB*/
	double *aval, *bval;		/* values in MatA and MatB*/

	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

	if (numtasks < 2 ) {
		fprintf(stderr, "err: more than 2 processes required\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(1);
	}
	numworkers = numtasks-1;

	/*============================================================================
				Master
	============================================================================*/
	if (taskid == MASTER){
		if(argc != 3){
			fprintf(stderr, "usage: %s mat1 mat2\n", argv[0]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}

		//Preprocessing of input files (idealy parallelised or done externally)
		char com1[100];
		char com2[100];
		sprintf(com1, "wc -l < %s; sort -k1 -n %s", argv[1], argv[1]);
		sprintf(com2, "wc -l < %s; sort -k2 -n %s", argv[2], argv[2]);

		//init file pointers
		FILE *fpa, *fpb, *fpout;
		fpa = popen(com1, "r");
		fpb = popen(com2, "r");

		//read file lengths files
		char buff[255];
		if(fgets(buff, 255, fpa) != NULL){
			alines = atoi(buff);
		}else{
			fprintf(stderr, "file %s unreadable\n", argv[1]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}

		if(fgets(buff, 255, fpb) != NULL){
			blines = atoi(buff);
		}else{
			fprintf(stderr, "err: file \"%s\" unreadable\n", argv[2]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}

		//allocate memory
		ay 	 = 	malloc(alines * sizeof(int));
		ax 	 = 	malloc(alines * sizeof(int));
		aval = 	malloc(alines * sizeof(double));
		by 	 = 	malloc(blines * sizeof(int));
		bx 	 = 	malloc(blines * sizeof(int));
		bval = 	malloc(blines * sizeof(double));

		if(ay==NULL || ax==NULL || aval==NULL || by==NULL || bx==NULL || bval==NULL){
			fprintf(stderr, "err: memory allocation error\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}

		//read content
		int aindex = 0, bindex = 0;
		while(fgets(buff, 255, fpa) != NULL){
			sscanf(buff, "%d %d %lf", &ay[aindex], &ax[aindex], &aval[aindex]);
			aindex++;
		}

		while(fgets(buff, 255, fpb) != NULL){
			sscanf(buff, "%d %d %lf", &by[bindex], &bx[bindex], &bval[bindex]);
			bindex++;
		}

		//locate partitions (segment largest matrix)
		int highest, next, ave, rem, chunk, breaks[numworkers + 1];
		chunk = 0;
		next = 1;

		highest = (alines >= blines) ? ay[alines - 1]: bx[blines - 1];
		if(highest < numworkers){
			fprintf(stderr, "err: cannot split %d rows between %d workers\n", highest, numworkers);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}

		ave = highest / numworkers;
		rem = highest % numworkers;

		for(int i = 0; i < ((alines >= blines) ? alines : blines); i++){
			if(((alines >= blines) ? ay[i] : bx[i]) >= next){
				next += (chunk < rem) ? ave+1 : ave;
				breaks[chunk++] = i;
			}
		}
		breaks[numworkers] = (alines >= blines) ? alines : blines;

		/* Send matrix data to the worker tasks */
		int rows;
		mtype = FROM_MASTER;
		for (dest=1; dest<=numworkers; dest++){
			rows = breaks[dest] - breaks[dest-1];
			if(alines > blines){
				//send matrix lengths
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&blines, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				//send matrix data
				MPI_Send(&ay[breaks[dest-1]], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ax[breaks[dest-1]], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send( by, blines, MPI_INT, dest,	mtype, MPI_COMM_WORLD);
				MPI_Send( bx, blines, MPI_INT, dest,	mtype, MPI_COMM_WORLD);
				MPI_Send(&aval[breaks[dest-1]], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send( bval, blines, MPI_DOUBLE, dest,	mtype, MPI_COMM_WORLD);
			}else{
				//send matrix lengths
				MPI_Send(&alines, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				//send matrix data
				MPI_Send( ay, alines, MPI_INT, dest,	mtype, MPI_COMM_WORLD);
				MPI_Send( ax, alines, MPI_INT, dest,	mtype, MPI_COMM_WORLD);
				MPI_Send(&by[breaks[dest-1]], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&bx[breaks[dest-1]], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send( aval, alines, MPI_DOUBLE, dest,	mtype, MPI_COMM_WORLD);
				MPI_Send(&bval[breaks[dest-1]], rows, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
			}
		}

		//Receive results from worker tasks
		int incount, *iny, *inx;
		double *inval;
		mtype = FROM_WORKER;

		for (source=1; source<=numworkers; source++){
			//recieve length of worker output
			MPI_Recv(&incount, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

			//allocate memory
			iny = malloc(incount * sizeof(int));
			inx = malloc(incount * sizeof(int));
			inval = malloc(incount * sizeof(double));

			//recieve data from worker
			MPI_Recv(iny, incount, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(inx, incount, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(inval, incount, MPI_DOUBLE,	source, mtype, MPI_COMM_WORLD, &status);

			//write output to stdout
			for(int j = 0; j < incount; j++){
				printf("%d %d %lf\n", iny[j], inx[j], inval[j]);
			}

			//free memory
			free(iny); free(inx); free(inval);
		}

		//close files
		fclose(fpa);
		fclose(fpb);
	}


	/*============================================================================
				Worker
	============================================================================*/
	if (taskid > MASTER){
		mtype = FROM_MASTER;
		//receive lengths
		MPI_Recv(&alines, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&blines, 1, MPI_INT, MASTER,	mtype, MPI_COMM_WORLD, &status);

		//allocate memory
		ay 	 = 	malloc(alines * sizeof(int));
		ax 	 = 	malloc(alines * sizeof(int));
		aval = 	malloc(alines * sizeof(double));
		by 	 = 	malloc(blines * sizeof(int));
		bx 	 = 	malloc(blines * sizeof(int));
		bval = 	malloc(blines * sizeof(double));

		//recieve data
		MPI_Recv(ay, alines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(ax, alines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(by, blines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(bx, blines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(aval, alines, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(bval, blines, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

		//evaluate product
		int outlines = 0;
		int outy[MAX_LINES], outx[MAX_LINES];
		double outval[MAX_LINES];

		#pragma omp parallel shared(outlines)
		{
			int mylines = 0;
			int myouty[MAX_LINES], myoutx[MAX_LINES];
			double myoutval[MAX_LINES];

			#pragma omp for collapse(2)
				for(int i = 0; i < alines; i++){
					for(int j = 0; j < blines; j++){
						if(ax[i] == by[j]){
							myouty[mylines] = ay[i];
							myoutx[mylines] = bx[j];
							myoutval[mylines] = aval[i] * bval[j];
							mylines++;
						}
					}
				}

			#pragma omp critical
			{
				//appends output lines - sums where needed
				int pairExists;
				for(int i = 0; i < mylines; i++){
					pairExists = 0;
					for(int j = 0; j < outlines; j++){
						if(outy[j] == myouty[i] && outx[j] == myoutx[i]){
							outval[j] += myoutval[i];
							pairExists = 1;
							break;
						}
					}
					if(!pairExists){
						outy[outlines] = myouty[i];
						outx[outlines] = myoutx[i];
						outval[outlines] = myoutval[i];
						outlines++;
					}
				}
			}
		}

		//send results
		mtype = FROM_WORKER;
		MPI_Send(&outlines, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send( outy, outlines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send( outx, outlines, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send( outval, outlines, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
	}

	//free memory
	free(ay); free(ax); free(aval);
	free(by);	free(bx);	free(bval);

	MPI_Finalize();
}
