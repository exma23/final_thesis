
void pllEvaluateLikelihood (pllInstance *tr, partitionList *pr, nodeptr p, pllBoolean fullTraversal, pllBoolean getPerSiteLikelihoods)
{
  /* now this may be the entry point of the library to compute
     the log like at a branch defined by p and p->back == q */

  volatile double
    result = 0.0;

  nodeptr
    q = p->back;


  pllBoolean
        p_recom = PLL_FALSE, /* if one of was missing, we will need to force recomputation */
        q_recom = PLL_FALSE;

  int
    i,
    model,
    numBranches = pr->perGeneBranchLengths?pr->numberOfPartitions : 1;

  /* if evaluate shall return the per-site log likelihoods
     fastScaling needs to be disabled, otherwise this will
     not work */

  if(getPerSiteLikelihoods)
    assert(!(tr->fastScaling));

  /* set the first entry of the traversal descriptor to contain the indices
     of nodes p and q */

  tr->td[0].ti[0].pNumber = p->number;
  tr->td[0].ti[0].qNumber = q->number;

  /* copy the branch lengths of the tree into the first entry of the traversal descriptor.
     if -M is not used tr->numBranches must be 1 */

  for(i = 0; i < numBranches; i++)
    tr->td[0].ti[0].qz[i] =  q->z[i];

  /* recom part */
  if(tr->useRecom)
  {
    int slot = -1;
    if(!isTip(q->number, tr->mxtips))
    {
      q_recom = getxVector(tr->rvec, q->number, &slot, tr->mxtips);
      tr->td[0].ti[0].slot_q = slot;
    }
    if(!isTip(p->number, tr->mxtips))
    {
      p_recom = getxVector(tr->rvec, p->number, &slot, tr->mxtips);
      tr->td[0].ti[0].slot_p = slot;
    }
    if(!isTip(p->number, tr->mxtips) &&  !isTip(q->number, tr->mxtips))
      assert(tr->td[0].ti[0].slot_q != tr->td[0].ti[0].slot_p);
  }


  /* now compute how many conditionals must be re-computed/re-oriented by newview
     to be able to calculate the likelihood at the root defined by p and q.
     */

  /* one entry in the traversal descriptor is already used, hence set the tarversal length counter to 1 */
  tr->td[0].count = 1;

  if(fullTraversal)
  {
    assert(isTip(q->back->number, tr->mxtips));
    computeTraversal(tr, q, PLL_FALSE, numBranches);
  }
  else
  {
    if(p_recom || needsRecomp(tr->useRecom, tr->rvec, p, tr->mxtips))
      computeTraversal(tr, p, PLL_TRUE, numBranches);

    if(q_recom || needsRecomp(tr->useRecom, tr->rvec, q, tr->mxtips))
      computeTraversal(tr, q, PLL_TRUE, numBranches);
  }


  /* now we copy this partition execute mask into the traversal descriptor which must come from the
     calling program, the logic of this should not form part of the library */

  storeExecuteMaskInTraversalDescriptor(tr, pr);

  /* also store in the traversal descriptor that something has changed i.e., in the parallel case that the
     traversal descriptor list of nodes needs to be broadcast once again */

  tr->td[0].traversalHasChanged = PLL_TRUE;
#if (defined(_FINE_GRAIN_MPI) || defined(_USE_PTHREADS))

  /* now here we enter the fork-join region for Pthreads */


  /* start the parallel region and tell all threads to compute the log likelihood for
     their fraction of the data. This call is implemented in the case switch of execFunction in axml.c
     */

  if(getPerSiteLikelihoods)
    {
      memset(tr->lhs, 0, sizeof(double) * tr->originalCrunchedLength);
      pllMasterBarrier(tr, pr, PLL_THREAD_EVALUATE_PER_SITE_LIKES);
    }
  else
    pllMasterBarrier (tr, pr, PLL_THREAD_EVALUATE);

  /* and now here we explicitly do the reduction operation , that is add over the
     per-thread and per-partition log likelihoods to obtain the overall log like
     over all sites and partitions */


  /*
     for unpartitioned data that's easy, we just sum over the log likes computed
     by each thread, thread 0 stores his results in reductionBuffer[0] thread 1 in
     reductionBuffer[1] and so on
     */

  /* This reduction for the partitioned case is more complicated because each thread
     needs to store the partial log like of each partition and we then need to collect
     and add everything */

#else
  /* and here is just the sequential case, we directly call pllEvaluateIterative() above
     without having to tell the threads/processes that they need to compute this function now */

  pllEvaluateIterative(tr, pr, getPerSiteLikelihoods); //PLL_TRUE

  /*
    if we want to obtain per-site rates they have initially been stored
     in arrays that are associated to the partition, now we
     copy them into the vector tr->lhs[].
     We may also chose that the user needs to rpovide an array, but this can be decided later-on.
  */

  if(getPerSiteLikelihoods) //PLL_TRUE
    {
      for(model = 0; model < pr->numberOfPartitions; model++)
        memcpy(&(tr->lhs[pr->partitionData[model]->lower]), pr->partitionData[model]->perSiteLikelihoods, pr->partitionData[model]->width  * sizeof(double));
    }

#endif

  for(model = 0; model < pr->numberOfPartitions; model++)
    result += pr->partitionData[model]->partitionLH;

  /* set the tree data structure likelihood value to the total likelihood */

  tr->likelihood = result;

  /* the code below is mainly for testing if the per-site log
     likelihoods we have stored in tr->lhs yield the same
     likelihood as the likelihood we computed.
     For numerical reasons we need to make a dirt PLL_ABS(difference) < epsilon
     comparison */

  if(getPerSiteLikelihoods) //PLL_TRUE
    {
      double
        likelihood = 0;
      int i;

      /* note that in tr->lhs, we just store the likelihood of
         one representative of a potentially compressed pattern,
         hence, we need to multiply the elemnts with the pattern
         weight vector */


      for(i = 0; i < tr->originalCrunchedLength; i++)
        {
//          printf("lhs[%d]=%f * %d\n", i, tr->lhs[i], tr->aliaswgt[i]);
          likelihood += (tr->lhs[i]   * tr->aliaswgt[i] );
        }

      if( PLL_ABS(tr->likelihood - likelihood) > 0.00001)
        {
  //        printf("likelihood was %f\t summed/weighted per-site-lnl was %f\n", tr->likelihood, likelihood);
        }

        assert(PLL_ABS(tr->likelihood - likelihood) < 0.00001);
    }


  if(tr->useRecom)
  {
    unpinNode(tr->rvec, p->number, tr->mxtips);
    unpinNode(tr->rvec, q->number, tr->mxtips);
  }

  /* do some bookkeeping to have traversalHasChanged in a consistent state */

  tr->td[0].traversalHasChanged = PLL_FALSE;
}
