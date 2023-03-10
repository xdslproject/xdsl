#include <stdio.h>
#include <mpi.h>

#define PRINT_GLOBAL_VAR(name) (printf("\t" #name " = %ld,\n", (long) name))

#define PRINT_GLOBAL_VAR_SIZE(name) (printf("\t" #name "_size = %ld,\n", sizeof(name)))

#define PRINT_STRUCT_FIELD_OFFSET(struct, name, field) (printf("\t" #name "_field_" #field " = %lld,    \t# offset of field " #field " in struct " #name "\n", (((long long int) &(struct.field)) - (long long int) &struct)))

void main() {

    printf("info = MpiLibraryInfo(\n");

    // Datatype
    printf("\t# MPI_Datatype\n");
    PRINT_GLOBAL_VAR_SIZE(MPI_Datatype);
    PRINT_GLOBAL_VAR(MPI_CHAR);
    PRINT_GLOBAL_VAR(MPI_SIGNED_CHAR);
    PRINT_GLOBAL_VAR(MPI_UNSIGNED_CHAR);
    PRINT_GLOBAL_VAR(MPI_BYTE);
    PRINT_GLOBAL_VAR(MPI_WCHAR);
    PRINT_GLOBAL_VAR(MPI_SHORT);
    PRINT_GLOBAL_VAR(MPI_UNSIGNED_SHORT);
    PRINT_GLOBAL_VAR(MPI_INT);
    PRINT_GLOBAL_VAR(MPI_UNSIGNED);
    PRINT_GLOBAL_VAR(MPI_LONG);
    PRINT_GLOBAL_VAR(MPI_UNSIGNED_LONG);
    PRINT_GLOBAL_VAR(MPI_FLOAT);
    PRINT_GLOBAL_VAR(MPI_DOUBLE);
    PRINT_GLOBAL_VAR(MPI_LONG_DOUBLE);
    PRINT_GLOBAL_VAR(MPI_LONG_LONG_INT);
    PRINT_GLOBAL_VAR(MPI_UNSIGNED_LONG_LONG);
    PRINT_GLOBAL_VAR(MPI_LONG_LONG);

    // Collectives
    printf("\n\t# MPI_Op\n");
    PRINT_GLOBAL_VAR_SIZE(MPI_Op);
    PRINT_GLOBAL_VAR(MPI_MAX);
    PRINT_GLOBAL_VAR(MPI_MIN);
    PRINT_GLOBAL_VAR(MPI_SUM);
    PRINT_GLOBAL_VAR(MPI_PROD);
    PRINT_GLOBAL_VAR(MPI_LAND);
    PRINT_GLOBAL_VAR(MPI_BAND);
    PRINT_GLOBAL_VAR(MPI_LOR);
    PRINT_GLOBAL_VAR(MPI_BOR);
    PRINT_GLOBAL_VAR(MPI_LXOR);
    PRINT_GLOBAL_VAR(MPI_BXOR);
    PRINT_GLOBAL_VAR(MPI_MINLOC);
    PRINT_GLOBAL_VAR(MPI_MAXLOC);
    PRINT_GLOBAL_VAR(MPI_REPLACE);
    PRINT_GLOBAL_VAR(MPI_NO_OP);

    // Communicators:
    printf("\n\t# MPI_Comm\n");
    PRINT_GLOBAL_VAR_SIZE(MPI_Comm);
    PRINT_GLOBAL_VAR(MPI_COMM_WORLD);
    PRINT_GLOBAL_VAR(MPI_COMM_SELF);

    // Request
    printf("\n\t# MPI_Request\n");
    PRINT_GLOBAL_VAR_SIZE(MPI_Request);

    // Status
    printf("\n\t# MPI_Status\n");
    PRINT_GLOBAL_VAR_SIZE(MPI_Status);
    PRINT_GLOBAL_VAR(MPI_STATUS_IGNORE);
    PRINT_GLOBAL_VAR(MPI_STATUSES_IGNORE);

    MPI_Status status;
    
    PRINT_STRUCT_FIELD_OFFSET(status, MPI_Status, MPI_SOURCE);
    PRINT_STRUCT_FIELD_OFFSET(status, MPI_Status, MPI_TAG);
    PRINT_STRUCT_FIELD_OFFSET(status, MPI_Status, MPI_ERROR);

    printf(")\n");
}
