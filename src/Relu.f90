!THE MODULE PROVIDES THE COLLECTION OF ACTIVATION FUNCTIONS FOR DEEP LEARING STUDIES.
!WRITTEN BY KALITH M ISMAIL, SEPTEMBER-2023, CFM, UPV-EHU, SPAIN.

    MODULE ReLU

        USE CUDAFOR

        USE TYPEX

        CONTAINS

        SUBROUTINE RELUFORWARD(LOCAL_BATCH)
    
            REAL(8), INTENT(INOUT) :: LOCAL_BATCH(:,:,:,:)
    
            REAL(8), ALLOCATABLE, DEVICE :: ACTX(:,:,:)
            INTEGER(4), DEVICE :: SZ_D(4)

            REAL(8), ALLOCATABLE :: ACT_X(:,:,:)

            INTEGER(4) :: I_BATCH
            INTEGER(4) :: NUM_THREADS, THREAD_ID
    
            INTEGER(4) :: SZ(4)
    
            TYPE(DIM3X) :: gridDim
            TYPE(DIM3X) :: blockDim

            SZ = SHAPE(LOCAL_BATCH)

            !Allocate the size of CUDA kernel
            ALLOCATE(ACTX(SZ(2), SZ(3), SZ(4)))
            ALLOCATE(ACT_X(SZ(2), SZ(3), SZ(4)))

            ! Calculate the appropriate CUDA grid and block dimensions
            blockDim = DIM3X(8, 8, 8)
            gridDim  = DIM3X(CEILING(REAL(SZ(2))/blockDim%x), CEILING(REAL(SZ(3))/blockDim%y), & 
                             CEILING(REAL(SZ(4))/blockDim%z))

            SZ_D = SZ

            ! Get the number of threads available in the OpenMP region
            NUM_THREADS = OMP_GET_MAX_THREADS()

            ! Perform batchwise computation on LOCAL_ARRAY with OpenMP threads
            !$OMP PARALLEL PRIVATE(I_BATCH, ACT_X, ACTX, THREAD_ID) SHARED(LOCAL_BATCH, SZ, SZ_D, blockDim, gridDim)
    
            ! Assign each thread to specific NUMA node memory based on thread ID
            THREAD_ID = OMP_GET_THREAD_NUM()

            ! Batch processing with NUMA-aware memory access
            !$OMP DO
            DO I_BATCH = 1, SZ(1)
                ! Perform some computation on LOCAL_BATCH, e.g., setting it to zero
                ACT_X = LOCAL_BATCH(I_BATCH,:,:,:)
                ACTX  = ACT_X
                
                ! Launch CUDA kernel with appropriate grid and block dimensions
                CALL RELUFORWARD_CUDA<<<gridDim, blockDim>>>(ACTX, SZ_D)

                ACT_X  = ACTX
                LOCAL_BATCH(I_BATCH,:,:,:) = ACT_X
            END DO
            !$OMP END DO

            !$OMP END PARALLEL

            ! Ensure that only one thread deallocates the memory
            ! Deallocate memory for the 4D array
            !$OMP MASTER
            DEALLOCATE(ACTX)
            DEALLOCATE(ACT_X)
            !$OMP END MASTER

            RETURN

        END SUBROUTINE RELUFORWARD

        SUBROUTINE RELUBACKWARD(LOCAL_BATCH)
    
            REAL(8), INTENT(INOUT) :: LOCAL_BATCH(:,:,:,:)
    
            REAL(8), ALLOCATABLE, DEVICE :: ACTDX(:,:,:)
            INTEGER(4), DEVICE :: SZ_D(4)

            REAL(8), ALLOCATABLE :: ACT_DX(:,:,:)

            INTEGER(4) :: I_BATCH
            INTEGER(4) :: NUM_THREADS, THREAD_ID
    
            INTEGER(4) :: SZ(4)
    
            TYPE(DIM3X) :: gridDim
            TYPE(DIM3X) :: blockDim

            SZ = SHAPE(LOCAL_BATCH)

            !Allocate the size of CUDA kernel
            ALLOCATE(ACTDX(SZ(2), SZ(3), SZ(4)))
            ALLOCATE(ACT_DX(SZ(2), SZ(3), SZ(4)))

            ! Calculate the appropriate CUDA grid and block dimensions
            blockDim = DIM3X(8, 8, 8)
            gridDim  = DIM3X(CEILING(REAL(SZ(2))/blockDim%x), CEILING(REAL(SZ(3))/blockDim%y), & 
                             CEILING(REAL(SZ(4))/blockDim%z))

            SZ_D = SZ

            ! Get the number of threads available in the OpenMP region
            NUM_THREADS = OMP_GET_MAX_THREADS()

            ! Perform batchwise computation on LOCAL_ARRAY with OpenMP threads
            !$OMP PARALLEL PRIVATE(I_BATCH, ACT_DX, ACTDX, THREAD_ID) SHARED(LOCAL_BATCH, SZ, SZ_D, blockDim, gridDim)
    
            ! Assign each thread to specific NUMA node memory based on thread ID
            THREAD_ID = OMP_GET_THREAD_NUM()

            ! Batch processing with NUMA-aware memory access
            !$OMP DO
            DO I_BATCH = 1, SZ(1)
                ! Perform some computation on LOCAL_BATCH, e.g., setting it to zero
                ACT_DX = LOCAL_BATCH(I_BATCH,:,:,:)
                ACTDX  = ACT_DX
                
                ! Launch CUDA kernel with appropriate grid and block dimensions
                CALL RELUBACKWARD_CUDA<<<gridDim, blockDim>>>(ACTDX, SZ_D)

                ACT_DX  = ACTDX
                LOCAL_BATCH(I_BATCH,:,:,:) = ACT_DX
            END DO
            !$OMP END DO

            !$OMP END PARALLEL

            ! Ensure that only one thread deallocates the memory
            ! Deallocate memory for the 4D array
            !$OMP MASTER
            DEALLOCATE(ACTDX)
            DEALLOCATE(ACT_DX)
            !$OMP END MASTER

            RETURN
        END SUBROUTINE RELUBACKWARD

    END MODULE ReLU