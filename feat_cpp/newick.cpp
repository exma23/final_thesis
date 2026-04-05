#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "pll.hpp"
#include "newick.hpp"
#include "lexer.hpp"
#include "mem_alloc.hpp"
#include "hash.hpp"

static int parse_newick (pllStack ** stack, int * inp)
{
  pllNewickNodeInfo * item = NULL;
  int item_active = 0;
  pllLexToken token;
  int input;
  pllLexToken prev_token;
  int nop = 0;          /* number of open parentheses */
  int depth = 0;

  prev_token.tokenType = PLL_TOKEN_UNKNOWN;

  input = *inp;

  NEXT_TOKEN

  while (token.tokenType != PLL_TOKEN_EOF && token.tokenType != PLL_TOKEN_UNKNOWN)
  {
    switch (token.tokenType)
     {
       case PLL_TOKEN_OPAREN:
        ++nop;
        memcpy (&prev_token, &token, sizeof (pllLexToken));
        ++depth;
        break;

       case PLL_TOKEN_CPAREN:
        if (prev_token.tokenType != PLL_TOKEN_CPAREN  &&
            prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
            prev_token.tokenType != PLL_TOKEN_STRING  &&
            prev_token.tokenType != PLL_TOKEN_NUMBER  &&
            prev_token.tokenType != PLL_TOKEN_FLOAT) return (0);

        if (!nop) return (0);
        --nop;
        memcpy (&prev_token, &token, sizeof (pllLexToken));

        /* push to the stack */
        if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo)); // possibly not nec
        //if (item->name   == NULL) item->name   = strdup ("INTERNAL_NODE");
        if (item->name == NULL)
         {
           item->name = (char *) rax_malloc ((strlen("INTERNAL_NODE") + 1) * sizeof (char));
           strcpy (item->name, "INTERNAL_NODE");
         }

        //if (item->branch == NULL) item->branch = strdup ("0.000000");
        if (item->branch == NULL)
         {
           item->branch = (char *) rax_malloc ((strlen("0.000000") + 1) * sizeof (char));
           strcpy (item->branch, "0.000000");
         }
        item->depth = depth;
        pllStackPush (stack, item);
        item_active  = 1;       /* active = 1 */
        item = NULL;
        --depth;
        break;

       case PLL_TOKEN_STRING:
        if (prev_token.tokenType != PLL_TOKEN_OPAREN &&
            prev_token.tokenType != PLL_TOKEN_CPAREN &&
            prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
            prev_token.tokenType != PLL_TOKEN_COMMA) return (0);
        if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo));
        //item->name = strndup (token.lexeme, token.len);
        item->name = (char *) rax_malloc ((token.len + 1) * sizeof (char));
        strncpy (item->name, token.lexeme, token.len);
        item->name[token.len] = 0;

        item_active = 1;
        item->depth = depth;
        if (prev_token.tokenType == PLL_TOKEN_COMMA  ||
            prev_token.tokenType == PLL_TOKEN_OPAREN ||
            prev_token.tokenType == PLL_TOKEN_UNKNOWN) item->leaf = 1;
        memcpy (&prev_token, &token, sizeof (pllLexToken));
        break;

       case PLL_TOKEN_FLOAT:
       case PLL_TOKEN_NUMBER:
         if  (prev_token.tokenType != PLL_TOKEN_OPAREN &&
              prev_token.tokenType != PLL_TOKEN_CPAREN &&
              prev_token.tokenType != PLL_TOKEN_COLON  &&
              prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
              prev_token.tokenType != PLL_TOKEN_COMMA) return (0);
        if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo));
        if (prev_token.tokenType == PLL_TOKEN_COLON)
         {
           //item->branch = strndup (token.lexeme, token.len);
           item->branch = (char *) rax_malloc ((token.len + 1) * sizeof (char));
           strncpy (item->branch, token.lexeme, token.len);
           item->branch[token.len] = 0;
         }
        else
         {
           if (prev_token.tokenType == PLL_TOKEN_COMMA  ||
               prev_token.tokenType == PLL_TOKEN_OPAREN ||
               prev_token.tokenType == PLL_TOKEN_UNKNOWN) item->leaf = 1;
           //if (prev_token.tokenType != PLL_TOKEN_UNKNOWN) ++ indent;
           //item->name = strndup (token.lexeme, token.len);
           item->name = (char *) rax_malloc ((token.len + 1) * sizeof (char));
           strncpy (item->name, token.lexeme, token.len);
           item->name[token.len] = 0;
         }
        item_active = 1;
        item->depth = depth;
        memcpy (&prev_token, &token, sizeof (pllLexToken));
        break;

       case PLL_TOKEN_COLON:
#ifdef PLLDEBUG
       printf ("PLL_TOKEN_COLON\n");
#endif
        if (prev_token.tokenType != PLL_TOKEN_CPAREN &&
            prev_token.tokenType != PLL_TOKEN_STRING &&
            prev_token.tokenType != PLL_TOKEN_FLOAT  &&
            prev_token.tokenType != PLL_TOKEN_NUMBER) return (0);
        memcpy (&prev_token, &token, sizeof (pllLexToken));
        break;

       case PLL_TOKEN_COMMA:
#ifdef PLLDEBUG
       printf ("PLL_TOKEN_COMMA\n");
#endif
        if (prev_token.tokenType != PLL_TOKEN_CPAREN &&
             prev_token.tokenType != PLL_TOKEN_STRING &&
             prev_token.tokenType != PLL_TOKEN_FLOAT &&
             prev_token.tokenType != PLL_TOKEN_NUMBER) return (0);
        memcpy (&prev_token, &token, sizeof (pllLexToken));

        /* push to the stack */
        if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo)); // possibly not nece
        //if (item->name   == NULL) item->name   = strdup ("INTERNAL_NODE");
        if (item->name == NULL)
         {
           item->name = (char *) rax_malloc ((strlen("INTERNAL_NODE") + 1) * sizeof (char));
           strcpy (item->name, "INTERNAL_NODE");
         }
        //if (item->branch == NULL) item->branch = strdup ("0.000000");
        if (item->branch == NULL)
         {
           item->branch = (char *) rax_malloc ((strlen("0.000000") + 1) * sizeof (char));
           strcpy (item->branch, "0.000000");
         }
        item->depth = depth;
        pllStackPush (stack, item);
        item_active  = 0;
        item = NULL;
        break;

       case PLL_TOKEN_SEMICOLON:
#ifdef PLLDEBUG
        printf ("PLL_TOKEN_SEMICOLON\n");
#endif
        /* push to the stack */
        if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo));
        //if (item->name   == NULL) item->name   = strdup ("ROOT_NODE");
        if (item->name == NULL)
         {
           item->name = (char *) rax_malloc ((strlen("ROOT_NODE") + 1) * sizeof (char));
           strcpy (item->name, "ROOT_NODE");
         }
        //if (item->branch == NULL) item->branch = strdup ("0.000000");
        if (item->branch == NULL)
         {
           item->branch = (char *) rax_malloc ((strlen("0.000000") + 1) * sizeof (char));
           strcpy (item->branch, "0.000000");
         }
        pllStackPush (stack, item);
        item_active  = 0;
        item = NULL;
        break;
       default:
       // TODO: Finish this part and add error codes
        break;
     }
    NEXT_TOKEN
    CONSUME(PLL_TOKEN_WHITESPACE | PLL_TOKEN_NEWLINE);
  }
  if (item_active)
   {
     if (!item) item = (pllNewickNodeInfo *) rax_calloc (1, sizeof (pllNewickNodeInfo));
     //if (item->name   == NULL) item->name   = strdup ("ROOT_NODE");
     if (item->name == NULL)
      {
        item->name = (char *) rax_malloc ((strlen("ROOT_NODE") + 1) * sizeof (char));
        strcpy (item->name, "ROOT_NODE");
      }
     //if (item->branch == NULL) item->branch = strdup ("0.000000");
     if (item->branch == NULL)
      {
        item->branch = (char *) rax_malloc ((strlen("0.000000") + 1) * sizeof (char));
        strcpy (item->branch, "0.000000");
      }
     pllStackPush (stack, item);
     item_active  = 0;
   }

  if (nop || token.tokenType == PLL_TOKEN_UNKNOWN)
   {
     return (0);
   }

  return (1);
}

static void assign_ranks (pllStack * stack, int * nodes, int * leaves)
{
  pllStack * head;
  pllNewickNodeInfo * item, * tmp;
  pllStack * preorder = NULL;
  int children;
  int depth;

  *nodes = *leaves = 0;


  head = stack;
  while (head)
  {
    assert (head->item);
    item = (pllNewickNodeInfo *) head->item;

    if (item->leaf)  ++ (*leaves);

    if (preorder)
     {
       tmp = (pllNewickNodeInfo *) preorder->item;
       children = 0;
       while (item->depth < tmp->depth)
        {
          children = 1;
          depth = tmp->depth;
          pllStackPop (&preorder);
          tmp = (pllNewickNodeInfo *) preorder->item;
          while (tmp->depth == depth)
           {
             ++ children;
             pllStackPop (&preorder);
             tmp = (pllNewickNodeInfo *)preorder->item;
           }
          tmp->rank += children;
        }
     }

    ++ (*nodes);
    head = head->next;

    if (item->leaf)
     {
       if (!preorder) return;

       children = 1;
       tmp = (pllNewickNodeInfo *) preorder->item;
       while (tmp->depth == item->depth)
        {
          ++ children;
          pllStackPop (&preorder);
          assert (preorder);
          tmp = (pllNewickNodeInfo *)preorder->item;
        }
       tmp->rank += children;
     }
    else
     {
       pllStackPush (&preorder, item);
     }
  }

  while (preorder->item != stack->item)
  {
    item = (pllNewickNodeInfo *)pllStackPop (&preorder);
    tmp  = (pllNewickNodeInfo *) preorder->item;
    children = 1;

    while (tmp->depth == item->depth)
     {
       ++ children;
       item = (pllNewickNodeInfo *) pllStackPop (&preorder);
       tmp  = (pllNewickNodeInfo *) preorder->item;
     }
    tmp->rank += children;
    children = 0;
  }
 assert (preorder->item == stack->item);

 pllStackClear (&preorder);
}

pllNewickTree * pllNewickParseString (const char * newick)
{
  int n, input, rc;
  pllNewickTree * t;
  int nodes, leaves;

  t = (pllNewickTree *) rax_calloc (1, sizeof (pllNewickTree));

  n = strlen (newick);


  init_lexan (newick, n);
  input = get_next_symbol();

  rc = parse_newick (&(t->tree), &input);
  if (!rc)
   {
     /* TODO: properly clean t->tree */
     rax_free (t);
     t = NULL;
   }
  else
   {
     assign_ranks (t->tree, &nodes, &leaves);
     t->nodes = nodes;
     t->tips  = leaves;
   }

  return (t);
}

void pllNewickParseDestroy (pllNewickTree ** t)
{
  pllNewickNodeInfo *  item;

  while ((item = (pllNewickNodeInfo *)pllStackPop (&((*t)->tree))))
   {
     rax_free (item->name);
     rax_free (item->branch);
     rax_free (item);
   }
  rax_free (*t);
  (*t) = NULL;
}