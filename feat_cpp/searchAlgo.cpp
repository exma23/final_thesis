/* ============================================================
 * Phiên bản đơn giản: liệt kê tất cả SPR moves, không đánh giá
 * Đặt vào: pllrepo/src/searchAlgo.c
 * Khai báo trong pll.h:
 *   pllRearrangeList * pllGetAllSPRMoves(pllInstance *tr, partitionList *pr, nodeptr p, int mintrav, int maxtrav, int maxMoves);
 * ============================================================ */

/* Hàm nội bộ: ghi 1 SPR move vào danh sách (không cần score) */
#include "searchAlgo.hpp"
#include <cstring>

static void pllCollectSPRMove (pllRearrangeList * list, nodeptr p, nodeptr q)
{
  pllRearrangeInfo rearr;
  int i;

  rearr.rearrangeType  = PLL_REARRANGE_SPR;
  rearr.likelihood     = 0.0;   /* không dùng */
  rearr.SPR.removeNode = p;
  rearr.SPR.insertNode = q;
  for (i = 0; i < PLL_NUM_BRANCHES; ++ i)
    rearr.SPR.zqr[i] = 0.0;    /* không dùng */

  if (list->entries < list->max_entries)
   {
     memcpy (&(list->rearr[list->entries]), &rearr, sizeof (pllRearrangeInfo));
     ++ list->entries;
   }
}

/* Hàm nội bộ: đệ quy duyệt cạnh trong khoảng [mintrav, maxtrav] */
static void pllTraverseCollect (pllInstance * tr, nodeptr p, nodeptr q, int mintrav, int maxtrav, pllRearrangeList * list)
{
  if (--mintrav <= 0)
    pllCollectSPRMove (list, p, q);

  if ((!isTip(q->number, tr->mxtips)) && (--maxtrav > 0))
   {
     pllTraverseCollect (tr, p, q->next->back,       mintrav, maxtrav, list);
     pllTraverseCollect (tr, p, q->next->next->back, mintrav, maxtrav, list);
   }
}

/* Hàm chính: với node p, prune 2 phía và thu thập tất cả vị trí chèn hợp lệ */
static int pllCollectSPR (pllInstance * tr, partitionList * pr, nodeptr p, int mintrav, int maxtrav, pllRearrangeList * list)
{
  nodeptr p1, p2, q, q1, q2;
  double  p1z[PLL_NUM_BRANCHES], p2z[PLL_NUM_BRANCHES];
  double  q1z[PLL_NUM_BRANCHES], q2z[PLL_NUM_BRANCHES];
  int     mintrav2, i;
  int     numBranches = pr->perGeneBranchLengths ? pr->numberOfPartitions : 1;

  if (maxtrav < 1 || mintrav > maxtrav) return PLL_FALSE;
  q = p->back;

  /* === Phía p: prune subtree tại p === */
  if (!isTip (p->number, tr->mxtips))
   {
     p1 = p->next->back;
     p2 = p->next->next->back;

     if (!isTip (p1->number, tr->mxtips) || !isTip (p2->number, tr->mxtips))
      {
        for (i = 0; i < numBranches; ++ i)
         { p1z[i] = p1->z[i]; p2z[i] = p2->z[i]; }

        /* Cắt p khỏi cây */
        removeNodeBIG (tr, pr, p, numBranches);

        /* Duyệt các vị trí chèn trên nhánh p1 */
        if (!isTip (p1->number, tr->mxtips))
         {
           pllTraverseCollect (tr, p, p1->next->back,       mintrav, maxtrav, list);
           pllTraverseCollect (tr, p, p1->next->next->back, mintrav, maxtrav, list);
         }
        /* Duyệt các vị trí chèn trên nhánh p2 */
        if (!isTip (p2->number, tr->mxtips))
         {
           pllTraverseCollect (tr, p, p2->next->back,       mintrav, maxtrav, list);
           pllTraverseCollect (tr, p, p2->next->next->back, mintrav, maxtrav, list);
         }

        /* Restore topology */
        hookup (p->next,       p1, p1z, numBranches);
        hookup (p->next->next, p2, p2z, numBranches);
      }
   }

  /* === Phía q (= p->back): prune subtree tại q === */
  if (!isTip (q->number, tr->mxtips) && maxtrav > 0)
   {
     q1 = q->next->back;
     q2 = q->next->next->back;

     if (
         (! isTip(q1->number, tr->mxtips) &&
          (! isTip(q1->next->back->number, tr->mxtips) || ! isTip(q1->next->next->back->number, tr->mxtips)))
         ||
         (! isTip(q2->number, tr->mxtips) &&
          (! isTip(q2->next->back->number, tr->mxtips) || ! isTip(q2->next->next->back->number, tr->mxtips)))
        )
      {
        for (i = 0; i < numBranches; ++ i)
         { q1z[i] = q1->z[i]; q2z[i] = q2->z[i]; }

        removeNodeBIG (tr, pr, q, numBranches);

        mintrav2 = mintrav > 2 ? mintrav : 2;

        if (!isTip (q1->number, tr->mxtips))
         {
           pllTraverseCollect (tr, q, q1->next->back,       mintrav2, maxtrav, list);
           pllTraverseCollect (tr, q, q1->next->next->back, mintrav2, maxtrav, list);
         }
        if (!isTip (q2->number, tr->mxtips))
         {
           pllTraverseCollect (tr, q, q2->next->back,       mintrav2, maxtrav, list);
           pllTraverseCollect (tr, q, q2->next->next->back, mintrav2, maxtrav, list);
         }

        hookup (q->next,       q1, q1z, numBranches);
        hookup (q->next->next, q2, q2z, numBranches);
      }
   }

  return PLL_TRUE;
}

/* API: liệt kê tất cả SPR moves trên toàn cây */
pllRearrangeList* pllGetAllSPRMoves (pllInstance * tr, partitionList * pr, int mintrav, int maxtrav, int maxMoves)
{
  pllRearrangeList * list;
  int i;

  list = pllCreateRearrangeList (maxMoves);

  for (i = 1; i <= tr->mxtips + tr->mxtips - 2; ++ i)
    pllCollectSPR (tr, pr, tr->nodep[i], mintrav, maxtrav, list);

  return list;
}