!THIS PACKAGE IS DEDICATED TO CREATE RANDOM NUMBER WITH GIVEN DISTRIBUTION.
!WRITTEN BY KALITH M ISMAIL, SEPTEMBER-2023, CFM, UPV-EHU, SPAIN.

MODULE RANDGEN

    USE TYPEX

    CONTAINS

        SUBROUTINE RND_BOXMULLER(X)

            REAL(8), INTENT(IN OUT) :: X(:,:,:)

            REAL(8), ALLOCATABLE, DEVICE :: RNDX(:,:,:)
    
            INTEGER :: SZ(3)
    
            TYPE(DIM3X) :: gridDim
            TYPE(DIM3X) :: blockDim

            SZ = SHAPE(X)

            ! Calculate the appropriate CUDA grid and block dimensions
            gridDim  = DIM3X(SZ(1), SZ(2), SZ(3))
            blockDim = DIM3X(1, 1, 1)

            !$OMP MASTER
            !Allocate the size of CUDA kernel
            ALLOCATE(RNDX(SZ(1), SZ(2), SZ(3)))

            CALL RANDOM_NUMBER(X(:,:,:))

            RNDX = X
        
            ! Launch CUDA kernel with appropriate grid and block dimensions
            CALL RND_BOXMULLER_CUDA<<<gridDim, blockDim>>>(RNDX)

            X = RNDX

            DEALLOCATE(RNDX)
            !$OMP END MASTER

        END SUBROUTINE RND_BOXMULLER

END MODULE RANDGEN
