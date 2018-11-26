/*
 * heatsim.c
 *
 *  Created on: 2011-11-17
 *      Author: francis
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include "config.h"
#include "part.h"
#include "grid.h"
#include "cart.h"
#include "image.h"
#include "heat.h"
#include "memory.h"
#include "util.h"

#define PROGNAME "heatsim"
#define DEFAULT_OUTPUT_PPM "heatsim.png"
#define DEFAULT_DIMX 1
#define DEFAULT_DIMY 1
#define DEFAULT_ITER 110
#define MAX_TEMP 1000.0
#define DIM_2D 2

typedef struct ctx {
	cart2d_t *cart;
	grid_t *global_grid;
	grid_t *curr_grid;
	grid_t *next_grid;
	grid_t *heat_grid;
	int numprocs;
	int rank;
	MPI_Comm comm2d;
	FILE *log;
	int verbose;
	int dims[DIM_2D];
	int isperiodic[DIM_2D];
	int coords[DIM_2D];
	int reorder;
	int north_peer;
	int south_peer;
	int east_peer;
	int west_peer;
	MPI_Datatype vector;
} ctx_t;

typedef struct command_opts {
	int dimx;
	int dimy;
	int iter;
	char *input;
	char *output;
	int verbose;
} opts_t;

static opts_t *global_opts = NULL;

__attribute__((noreturn))
static void usage(void) {
	fprintf(stderr, PROGNAME " " VERSION " " PACKAGE_NAME "\n");
	fprintf(stderr, "Usage: " PROGNAME " [OPTIONS] [COMMAND]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  --help	this help\n");
	fprintf(stderr, "  --iter	number of iterations to perform\n");
	fprintf(stderr, "  --dimx	2d decomposition in x dimension\n");
	fprintf(stderr, "  --dimy	2d decomposition in y dimension\n");
	fprintf(stderr, "  --input  png input file\n");
	fprintf(stderr, "  --output ppm output file\n");
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

static void dump_opts(struct command_opts *opts) {
	printf("%10s %s\n", "option", "value");
	printf("%10s %d\n", "dimx", opts->dimx);
	printf("%10s %d\n", "dimy", opts->dimy);
	printf("%10s %d\n", "iter", opts->iter);
	printf("%10s %s\n", "input", opts->input);
	printf("%10s %s\n", "output", opts->output);
	printf("%10s %d\n", "verbose", opts->verbose);
}

void default_int_value(int *val, int def) {
	if (*val == 0)
		*val = def;
}

static int parse_opts(int argc, char **argv, struct command_opts *opts) {
	int idx;
	int opt;
	int ret = 0;

	struct option options[] = { { "help", 0, 0, 'h' },
			{ "iter", 1, 0, 'r' }, { "dimx", 1, 0, 'x' }, { "dimy",
					1, 0, 'y' }, { "input", 1, 0, 'i' }, {
					"output", 1, 0, 'o' }, { "verbose", 0,
					0, 'v' }, { 0, 0, 0, 0 } };

	memset(opts, 0, sizeof(struct command_opts));

	while ((opt = getopt_long(argc, argv, "hvx:y:l:", options, &idx)) != -1) {
		switch (opt) {
		case 'r':
			opts->iter = atoi(optarg);
			break;
		case 'y':
			opts->dimy = atoi(optarg);
			break;
		case 'x':
			opts->dimx = atoi(optarg);
			break;
		case 'i':
			if (asprintf(&opts->input, "%s", optarg) < 0)
				goto err;
			break;
		case 'o':
			if (asprintf(&opts->output, "%s", optarg) < 0)
				goto err;
			break;
		case 'h':
			usage();
			break;
		case 'v':
			opts->verbose = 1;
			break;
		default:
			printf("unknown option %c\n", opt);
			ret = -1;
			break;
		}
	}

	/* default values*/
	default_int_value(&opts->iter, DEFAULT_ITER);
	default_int_value(&opts->dimx, DEFAULT_DIMX);
	default_int_value(&opts->dimy, DEFAULT_DIMY);
	if (opts->output == NULL)
		if (asprintf(&opts->output, "%s", DEFAULT_OUTPUT_PPM) < 0)
			goto err;
	if (opts->input == NULL) {
		fprintf(stderr, "missing input file");
		goto err;
	}

	if (opts->dimx == 0 || opts->dimy == 0) {
		fprintf(stderr,
				"argument error: dimx and dimy must be greater than 0\n");
		ret = -1;
	}

	if (opts->verbose)
		dump_opts(opts);
	global_opts = opts;
	return ret;
	err:
	FREE(opts->input);
	FREE(opts->output);
	return -1;
}

FILE *open_logfile(int rank) {
	char str[255];
	sprintf(str, "out-%d", rank);
	FILE *f = fopen(str, "w+");
	return f;
}

ctx_t *make_ctx() {
	ctx_t *ctx = (ctx_t *) calloc(1, sizeof(ctx_t));
	return ctx;
}

void free_ctx(ctx_t *ctx) {
	if (ctx == NULL)
		return;
	free_grid(ctx->global_grid);
	free_grid(ctx->curr_grid);
	free_grid(ctx->next_grid);
	free_grid(ctx->heat_grid);
	free_cart2d(ctx->cart);
	if (ctx->log != NULL) {
		fflush(ctx->log);
		fclose(ctx->log);
	}
	FREE(ctx);
}

int init_ctx(ctx_t *ctx, opts_t *opts) {
	MPI_Comm_size(MPI_COMM_WORLD, &ctx->numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);

	if (opts->dimx * opts->dimy != ctx->numprocs) {
		fprintf(stderr,
				"2D decomposition blocks must equal number of process\n");
		goto err;
	}

	ctx->log = open_logfile(ctx->rank);
	ctx->verbose = opts->verbose;
	ctx->dims[0] = opts->dimx;
	ctx->dims[1] = opts->dimy;
	ctx->isperiodic[0] = 1;
	ctx->isperiodic[1] = 1;
	ctx->reorder = 0;
	grid_t *new_grid = NULL;
	image_t *image;
	int res;
	int width, height, rank_tag;

	/* TODO: Créer un "2D cartesian communicator" */
	MPI_Cart_create(MPI_COMM_WORLD, DIM_2D, ctx->dims, ctx->isperiodic, ctx->reorder, &ctx->comm2d);

    // Find the peers
	MPI_Cart_shift(ctx->comm2d, 0, 1, &ctx->west_peer, &ctx->east_peer);
	MPI_Cart_shift(ctx->comm2d, 1, 1, &ctx->north_peer, &ctx->south_peer);

	MPI_Cart_coords(ctx->comm2d, ctx->rank, DIM_2D, ctx->coords);

	/*
	 * TODO: Le processus rank=0 charge l'image du disque
	 * et transfert chaque section aux autres processus
	 */
	if (ctx->rank == 0) {
		int coords[DIM_2D];
		/* Charger l'image d'entrée */
		image = load_png(opts->input);
		if (image == NULL)
			goto err;
		/* Initialisation de la grid avec le canal rouge */
		ctx->global_grid = grid_from_image(image, CHAN_RED);

		/* La grid a été normalisée à 1, mutiplication par MAX_TEMP */
		grid_multiply(ctx->global_grid, MAX_TEMP);

		/* Décomposition 2D */
		ctx->cart = make_cart2d(ctx->global_grid->width,
				ctx->global_grid->height, opts->dimx, opts->dimy);
		cart2d_grid_split(ctx->cart, ctx->global_grid);

		/*
		 * TODO: Envoyer les dimensions de la grid dimensions et les données
		 * Comment traiter le cas de rank=0 ?
		 */
		grid_t *grid;
		int rank;
		MPI_Request req[(ctx->numprocs - 1) * 3];
		MPI_Status status[(ctx->numprocs - 1) * 3];
		for (rank = 1; rank < ctx->numprocs; rank++) {
			MPI_Cart_coords(ctx->comm2d, rank, DIM_2D, coords);
			// Get the grid for current rank
			grid = cart2d_get_grid(ctx->cart, coords[0], coords[1]);
			// Send dimensions and data, using unique tags for each message
			rank_tag = rank * 3;
			res = MPI_Isend(&grid->width, 1, MPI_INTEGER, rank, rank_tag, ctx->comm2d, &req[(rank - 1) * 3]);
			res &= MPI_Isend(&grid->height, 1, MPI_INTEGER, rank, rank_tag + 1, ctx->comm2d, &req[(rank - 1) * 3 + 1]);
			res &= MPI_Isend(grid->data, grid->width * grid->height, MPI_INTEGER, rank, rank_tag + 2, ctx->comm2d, &req[(rank - 1) * 3 + 2]);
			if (res != MPI_SUCCESS)
				goto err;
		}
		MPI_Waitall((ctx->numprocs -1) * 3, req, status);
		MPI_Cart_coords(ctx->comm2d, ctx->rank, DIM_2D, coords);
		// Get the grid for rank 0 and stock it in new_grid
		grid = cart2d_get_grid(ctx->cart, coords[0], coords[1]);
		new_grid = make_grid(grid->width, grid->height, 0);
		grid_copy(grid, new_grid);
	} else {
		/*
	 	 * TODO: Recevoir les dimensions de la grid
	 	 * et stocker dans new_grid
	 	 */
		MPI_Request requests[3];
		MPI_Status status[3];

		rank_tag = ctx->rank * 3;

		res = MPI_Irecv(&width, 1, MPI_INTEGER, 0, rank_tag, ctx->comm2d, &requests[0]);
		res &= MPI_Irecv(&height, 1, MPI_INTEGER, 0, rank_tag + 1, ctx->comm2d, &requests[1]);
		// Before creating the grid, we must wait for all parameters to be received
		MPI_Waitall(2, requests, status);
		new_grid = make_grid(width, height, 0);

		res &= MPI_Irecv(new_grid->data, width * height, MPI_INTEGER, 0, rank_tag + 2, ctx->comm2d, &requests[2]);
		MPI_Wait(&requests[2], &status[2]);
	}

	if (new_grid == NULL)
		goto err;

	
	
	/* set padding required for Runge-Kutta */
	ctx->curr_grid = grid_padding(new_grid, 1);
	ctx->next_grid = grid_padding(new_grid, 1);
	ctx->heat_grid = grid_padding(new_grid, 1);
	free_grid(new_grid);

	/* TODO: Créer un type vector pour échanger les colonnes */
	MPI_Type_vector(ctx->curr_grid->height, 1, ctx->curr_grid->width, MPI_INTEGER, &ctx->vector);
	MPI_Type_commit(&ctx->vector);

	return 0;
	err: return -1;
}

void dump_ctx(ctx_t *ctx) {
	fprintf(ctx->log, "*** CONTEXT ***\n");
	fprintf(ctx->log, "rank=%d\n", ctx->rank);
	fprintf(ctx->log, "north=%d south=%d west=%d east=%d \n",
			ctx->north_peer, ctx->south_peer,
			ctx->east_peer, ctx->west_peer);
	fprintf(ctx->log, "***************\n");
}

void exchng2d(ctx_t *ctx) {
	/*
	 *  TODO: Échanger les bordures avec les voisins
	 * 4 échanges doivent etre effectués
	 */
	grid_t *grid = ctx->next_grid;
	int width = grid->pw;
	int height = grid->ph;
	int *data = grid->data;
	MPI_Comm comm = ctx->comm2d;
	int north_peer = ctx->north_peer;
	int south_peer = ctx->south_peer;
	int east_peer = ctx->east_peer;
	int west_peer = ctx->west_peer;
	MPI_Status status[4];

	//Calcul des offsets
    int *offset_send_north = data + width;
    int *offset_recv_north = data;

    int *offset_send_south = data + width * (height -  2);
    int *offset_recv_south = data + width * (height -  1);

	int *offset_send_east = data + width - 2;
    int *offset_recv_east = data + width + 1;

    int *offset_send_west = data + 1;
    int *offset_recv_west = data;

	// North to south
	MPI_Sendrecv(offset_send_south, width, MPI_INTEGER, south_peer, 0,
		offset_recv_south, width, MPI_INTEGER, north_peer, 0, comm, &status[0]);
	// South to north
	MPI_Sendrecv(offset_send_north, width, MPI_INTEGER, north_peer, 1,
		offset_recv_north, width, MPI_INTEGER, south_peer, 1, comm, &status[1]);
	// West to east
	MPI_Sendrecv(offset_send_east, 1, ctx->vector, east_peer, 2,
		offset_recv_east, 1, ctx->vector, west_peer, 2, comm, &status[2]);
	// East to west
	MPI_Sendrecv(offset_send_west, 1, ctx->vector, west_peer, 3,
		offset_recv_west, 1, ctx->vector, east_peer, 3, comm, &status[3]);
}

int gather_result(ctx_t *ctx, opts_t *opts) {
	int ret = 0;
	int res;
	grid_t *local_grid = grid_padding(ctx->next_grid, 0);
	if (local_grid == NULL)
		goto err;

	/*
	 * TODO: Transférer les résultats de la simulation vers le rank=0.
	 * Utiliser grid pour ceci.
	 */
	if (ctx->rank == 0) {
		int coords[DIM_2D];
		MPI_Request req[ctx->numprocs];
		MPI_Status status[ctx->numprocs];
		// Copy the grid of rank 0
		MPI_Cart_coords(ctx->comm2d, 0, DIM_2D, coords);
		grid_copy(local_grid, cart2d_get_grid(ctx->cart, coords[0], coords[1]));

		int rank;
		for (rank = 1; rank < ctx->numprocs; rank++) {
			MPI_Cart_coords(ctx->comm2d, rank, DIM_2D, coords);
			grid_t *grid = cart2d_get_grid(ctx->cart, coords[0], coords[1]);
			MPI_Irecv(grid->data, grid->width * grid->height, MPI_INTEGER, rank, 0, ctx->comm2d, &req[rank-1]);
		}
		MPI_Waitall(ctx->numprocs - 1, req, status);
		/* now we can merge all data blocks, reuse global_grid */
		cart2d_grid_merge(ctx->cart, ctx->global_grid);
	} else {
		res = MPI_Send(local_grid->data, local_grid->width * local_grid->height, MPI_INTEGER, 0, 0, ctx->comm2d);
		if (res != MPI_SUCCESS)
			goto err;
	}

	done: free_grid(local_grid);
	return ret;
	err: ret = -1;
	goto done;
}

int main(int argc, char **argv) {
	ctx_t *ctx = NULL;
	int rep, ret;
	opts_t opts;

	if (parse_opts(argc, argv, &opts) < 0) {
		printf("Error while parsing arguments\n");
		usage();
	}
	if (opts.verbose)
		dump_opts(&opts);

	MPI_Init(&argc, &argv);

	ctx = make_ctx();
	if (init_ctx(ctx, &opts) < 0)
		goto err;
	if (opts.verbose)
		dump_ctx(ctx);

	if (ctx->verbose) {
		fprintf(ctx->log, "heat grid\n");
		fdump_grid(ctx->heat_grid, ctx->log);
	}

	for (rep = 0; rep < opts.iter; rep++) {
		if (ctx->verbose) {
			fprintf(ctx->log, "iter %d\n", rep);
			fprintf(ctx->log, "start\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}

		grid_set_min(ctx->heat_grid, ctx->curr_grid);
		if (ctx->verbose) {
			fprintf(ctx->log, "grid_set_min\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}

		exchng2d(ctx);
		if (ctx->verbose) {
			fprintf(ctx->log, "exchng2d\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}

		heat_diffuse(ctx->curr_grid, ctx->next_grid);
		if (ctx->verbose) {
			fprintf(ctx->log, "heat_diffuse\n");
			fdump_grid(ctx->next_grid, ctx->log);
		}
		SWAP(ctx->curr_grid, ctx->next_grid);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (gather_result(ctx, &opts) < 0)
		goto err;

	if (ctx->rank == 0) {
		printf("saving...\n");
		if (save_grid_png(ctx->global_grid, opts.output) < 0) {
			printf("saving failed\n");
			goto err;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	ret = EXIT_SUCCESS;
done:
	free_ctx(ctx);
	MPI_Finalize();
	FREE(opts.input);
	FREE(opts.output);
	return ret;
err:
	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	ret = EXIT_FAILURE;
	goto done;
}

