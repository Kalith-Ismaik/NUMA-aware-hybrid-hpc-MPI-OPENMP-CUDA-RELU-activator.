PROGRAM MAIN

    USE RANDGEN

    IMPLICIT NONE

    INCLUDE 'mpif.h'

    INTEGER, PARAMETER :: NODES = 2, CORES_PER_NODE = 64
    INTEGER, PARAMETER :: GPU_PER_NODE = 2

    INTEGER :: rank, comm_size, ierr
    INTEGER :: istatus(MPI_STATUS_SIZE)

    INTEGER :: COUNT_START, COUNT_END, COUNT_RATE

    REAL(8), ALLOCATABLE :: INPUT(:,:,:,:)
    
    REAL(8) :: START, END

    INTEGER :: I
    INTEGER :: BATCH
    INTEGER :: LOCAL_BATCH, I_LOCAL

    CALL MPI_Init(ierr)
    CALL MPI_Comm_size(MPI_COMM_WORLD,comm_size,ierr)
    CALL MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

    ! Ensure the number of processes is compatible
    IF(comm_size /= NODES) THEN
        WRITE(*,*) "ERROR 101: Number of MPI processes must match number of nodes! Quitting..."
        CALL MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        STOP
    END IF

!-----------------------------------------------------

    ALLOCATE(INPUT(100,100,100,20000))

    BATCH = 20000
    LOCAL_BATCH = BATCH/NODES

!-----------------------------------------------------

    ! Scatter the INPUT array
    CALL MPI_Scatter(INPUT, LOCAL_BATCH*100*100*100, MPI_REAL8, INPUT, LOCAL_BATCH*100*100*100, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

!------------------------------------------------------

    ! Start timing
    CALL SYSTEM_CLOCK(COUNT_START, COUNT_RATE)
    START = COUNT_START / COUNT_RATE

    ! OpenMP parallel region for CPU processing
    !$OMP PARALLEL PRIVATE(I_LOCAL)
    !$OMP DO
    DO I_LOCAL = 1, LOCAL_BATCH
        CALL RND_BOXMULLER(INPUT(:,:,:,I_LOCAL))
    END DO
    !$OMP END PARALLEL

    ! End timing
    CALL SYSTEM_CLOCK(COUNT_END)
    END = COUNT_END / COUNT_RATE

    PRINT '("Time = ",f6.3," seconds.")', END - START

!------------------------------------------------------

    ! Gather results
    CALL MPI_Gather(INPUT, LOCAL_BATCH*100*100*100, MPI_REAL8, INPUT, LOCAL_BATCH*100*100*100, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

!------------------------------------------------------

    DEALLOCATE(INPUT)

    CALL MPI_Barrier(MPI_COMM_WORLD, ierr)
    CALL MPI_Finalize(ierr)

END PROGRAM MAIN
