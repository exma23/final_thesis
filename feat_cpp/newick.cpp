#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "errcodes.hpp"
#include "hash.hpp"
#include "lexer.hpp"
#include "mem_alloc.hpp"
#include "newick.hpp"
#include "pll.hpp"

static void free_newick_item(pllNewickNodeInfo *item) {
  if (!item)
    return;
  rax_free(item->name);
  rax_free(item->branch);
  rax_free(item);
}

// Parse chuỗi Newick đã được lexer đọc token, rồi đưa các node vào stack.
static int parse_newick(pllStack **stack, int *inp) {
  pllNewickNodeInfo *item = NULL;
  int item_active = 0;
  pllLexToken token;
  int input;
  pllLexToken prev_token;
  int nop = 0; /* number of open parentheses */
  int depth = 0;
  int saw_semicolon = 0;

  prev_token.tokenType = PLL_TOKEN_UNKNOWN;

  input = *inp;

  NEXT_TOKEN

  while (token.tokenType != PLL_TOKEN_EOF &&
         token.tokenType != PLL_TOKEN_UNKNOWN) {
    switch (token.tokenType) {
    case PLL_TOKEN_OPAREN:
      ++nop;
      memcpy(&prev_token, &token, sizeof(pllLexToken));
      ++depth;
      break;

    case PLL_TOKEN_CPAREN:
      if (prev_token.tokenType != PLL_TOKEN_CPAREN &&
          prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
          prev_token.tokenType != PLL_TOKEN_STRING &&
          prev_token.tokenType != PLL_TOKEN_NUMBER &&
          prev_token.tokenType != PLL_TOKEN_FLOAT)
        goto fail;

      if (!nop)
        goto fail;

      --nop;
      memcpy(&prev_token, &token, sizeof(pllLexToken));

      if (!item)
        item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

      if (!item)
        goto fail;

      if (item->name == NULL) {
        item->name =
            (char *)rax_malloc((strlen("INTERNAL_NODE") + 1) * sizeof(char));
        if (!item->name)
          goto fail;
        strcpy(item->name, "INTERNAL_NODE");
      }

      if (item->branch == NULL) {
        item->branch =
            (char *)rax_malloc((strlen("0.000000") + 1) * sizeof(char));
        if (!item->branch)
          goto fail;
        strcpy(item->branch, "0.000000");
      }

      item->depth = depth;
      if (!pllStackPush(stack, item))
        goto fail;

      item_active = 1;
      item = NULL;
      --depth;
      break;

    case PLL_TOKEN_STRING:
      if (prev_token.tokenType != PLL_TOKEN_OPAREN &&
          prev_token.tokenType != PLL_TOKEN_CPAREN &&
          prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
          prev_token.tokenType != PLL_TOKEN_COMMA)
        goto fail;

      if (!item)
        item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

      if (!item)
        goto fail;

      item->name = (char *)rax_malloc((token.len + 1) * sizeof(char));
      if (!item->name)
        goto fail;

      strncpy(item->name, token.lexeme, token.len);
      item->name[token.len] = 0;

      item_active = 1;
      item->depth = depth;

      if (prev_token.tokenType == PLL_TOKEN_COMMA ||
          prev_token.tokenType == PLL_TOKEN_OPAREN ||
          prev_token.tokenType == PLL_TOKEN_UNKNOWN)
        item->leaf = 1;

      memcpy(&prev_token, &token, sizeof(pllLexToken));
      break;

    case PLL_TOKEN_FLOAT:
    case PLL_TOKEN_NUMBER:
      if (prev_token.tokenType != PLL_TOKEN_OPAREN &&
          prev_token.tokenType != PLL_TOKEN_CPAREN &&
          prev_token.tokenType != PLL_TOKEN_COLON &&
          prev_token.tokenType != PLL_TOKEN_UNKNOWN &&
          prev_token.tokenType != PLL_TOKEN_COMMA)
        goto fail;

      if (!item)
        item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

      if (!item)
        goto fail;

      if (prev_token.tokenType == PLL_TOKEN_COLON) {
        item->branch = (char *)rax_malloc((token.len + 1) * sizeof(char));
        if (!item->branch)
          goto fail;

        strncpy(item->branch, token.lexeme, token.len);
        item->branch[token.len] = 0;
      } else {
        if (prev_token.tokenType == PLL_TOKEN_COMMA ||
            prev_token.tokenType == PLL_TOKEN_OPAREN ||
            prev_token.tokenType == PLL_TOKEN_UNKNOWN)
          item->leaf = 1;

        item->name = (char *)rax_malloc((token.len + 1) * sizeof(char));
        if (!item->name)
          goto fail;

        strncpy(item->name, token.lexeme, token.len);
        item->name[token.len] = 0;
      }

      item_active = 1;
      item->depth = depth;
      memcpy(&prev_token, &token, sizeof(pllLexToken));
      break;

    case PLL_TOKEN_COLON:
      if (prev_token.tokenType != PLL_TOKEN_CPAREN &&
          prev_token.tokenType != PLL_TOKEN_STRING &&
          prev_token.tokenType != PLL_TOKEN_FLOAT &&
          prev_token.tokenType != PLL_TOKEN_NUMBER)
        goto fail;

      memcpy(&prev_token, &token, sizeof(pllLexToken));
      break;

    case PLL_TOKEN_COMMA:
      if (prev_token.tokenType != PLL_TOKEN_CPAREN &&
          prev_token.tokenType != PLL_TOKEN_STRING &&
          prev_token.tokenType != PLL_TOKEN_FLOAT &&
          prev_token.tokenType != PLL_TOKEN_NUMBER)
        goto fail;

      memcpy(&prev_token, &token, sizeof(pllLexToken));

      if (!item)
        item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

      if (!item)
        goto fail;

      if (item->name == NULL) {
        item->name =
            (char *)rax_malloc((strlen("INTERNAL_NODE") + 1) * sizeof(char));
        if (!item->name)
          goto fail;
        strcpy(item->name, "INTERNAL_NODE");
      }

      if (item->branch == NULL) {
        item->branch =
            (char *)rax_malloc((strlen("0.000000") + 1) * sizeof(char));
        if (!item->branch)
          goto fail;
        strcpy(item->branch, "0.000000");
      }

      item->depth = depth;
      if (!pllStackPush(stack, item))
        goto fail;

      item_active = 0;
      item = NULL;
      break;

    case PLL_TOKEN_SEMICOLON:
      saw_semicolon = 1;

      if (!item)
        item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

      if (!item)
        goto fail;

      if (item->name == NULL) {
        item->name =
            (char *)rax_malloc((strlen("ROOT_NODE") + 1) * sizeof(char));
        if (!item->name)
          goto fail;
        strcpy(item->name, "ROOT_NODE");
      }

      if (item->branch == NULL) {
        item->branch =
            (char *)rax_malloc((strlen("0.000000") + 1) * sizeof(char));
        if (!item->branch)
          goto fail;
        strcpy(item->branch, "0.000000");
      }

      if (!pllStackPush(stack, item))
        goto fail;

      item_active = 0;
      item = NULL;
      break;

    default:
      goto fail;
    }

    NEXT_TOKEN
    CONSUME(PLL_TOKEN_WHITESPACE | PLL_TOKEN_NEWLINE);
  }

  if (!saw_semicolon)
    goto fail;

  if (item_active) {
    if (!item)
      item = (pllNewickNodeInfo *)rax_calloc(1, sizeof(pllNewickNodeInfo));

    if (!item)
      goto fail;

    if (item->name == NULL) {
      item->name = (char *)rax_malloc((strlen("ROOT_NODE") + 1) * sizeof(char));
      if (!item->name)
        goto fail;
      strcpy(item->name, "ROOT_NODE");
    }

    if (item->branch == NULL) {
      item->branch =
          (char *)rax_malloc((strlen("0.000000") + 1) * sizeof(char));
      if (!item->branch)
        goto fail;
      strcpy(item->branch, "0.000000");
    }

    if (!pllStackPush(stack, item))
      goto fail;

    item_active = 0;
    item = NULL;
  }

  if (nop || token.tokenType == PLL_TOKEN_UNKNOWN)
    goto fail;

  return (1);

fail:
  free_newick_item(item);
  return (0);
}

/*
  Duyệt stack đã parse để:
  - đếm tổng số node
  - đếm số leaf
  - tính rank cho internal nodes: rank la so luong leaves cua 1 node
*/
static void assign_ranks(pllStack *stack, int *nodes, int *leaves) {
  pllStack *head;
  pllNewickNodeInfo *item, *tmp;
  pllStack *preorder = NULL;
  int children;
  int depth;

  *nodes = *leaves = 0;

  if (!stack)
    return;

  head = stack;
  while (head) {
    assert(head->item);
    item = (pllNewickNodeInfo *)head->item;

    if (item->leaf)
      ++(*leaves);

    if (preorder) {
      tmp = (pllNewickNodeInfo *)preorder->item;
      children = 0;
      while (item->depth < tmp->depth) {
        children = 1;
        depth = tmp->depth;
        pllStackPop(&preorder);

        if (!preorder)
          break;

        tmp = (pllNewickNodeInfo *)preorder->item;
        while (preorder && tmp->depth == depth) {
          ++children;
          pllStackPop(&preorder);
          if (!preorder)
            break;
          tmp = (pllNewickNodeInfo *)preorder->item;
        }

        if (preorder)
          tmp->rank += children;
      }
    }

    ++(*nodes);
    head = head->next;

    if (item->leaf) {
      if (!preorder)
        return;

      children = 1;
      tmp = (pllNewickNodeInfo *)preorder->item;
      while (preorder && tmp->depth == item->depth) {
        ++children;
        pllStackPop(&preorder);
        assert(preorder);
        tmp = (pllNewickNodeInfo *)preorder->item;
      }
      tmp->rank += children;
    } else {
      pllStackPush(&preorder, item);
    }
  }

  if (!preorder)
    return;

  while (preorder && preorder->item != stack->item) {
    item = (pllNewickNodeInfo *)pllStackPop(&preorder);
    if (!preorder)
      break;

    tmp = (pllNewickNodeInfo *)preorder->item;
    children = 1;

    while (preorder && tmp->depth == item->depth) {
      ++children;
      item = (pllNewickNodeInfo *)pllStackPop(&preorder);
      if (!preorder)
        break;
      tmp = (pllNewickNodeInfo *)preorder->item;
    }

    if (preorder)
      tmp->rank += children;

    children = 0;
  }

  if (preorder)
    assert(preorder->item == stack->item);

  pllStackClear(&preorder);
}

pllNewickTree *pllNewickParseString(const char *newick) {
  int input, rc;
  pllNewickTree *t;
  int nodes, leaves;

  if (!newick)
    return NULL;

  t = (pllNewickTree *)rax_calloc(1, sizeof(pllNewickTree));
  if (!t)
    return NULL;

  init_lexan(newick, (long)strlen(newick));
  input = get_next_symbol();

  rc = parse_newick(&(t->tree), &input);
  if (!rc) {
    pllNewickParseDestroy(&t);
    return NULL;
  }

  assign_ranks(t->tree, &nodes, &leaves);
  t->nodes = nodes;
  t->tips = leaves;

  return (t);
}

void pllNewickParseDestroy(pllNewickTree **t) {
  pllNewickNodeInfo *item;

  if (!t || !*t)
    return;

  while ((item = (pllNewickNodeInfo *)pllStackPop(&((*t)->tree)))) {
    rax_free(item->name);
    rax_free(item->branch);
    rax_free(item);
  }

  rax_free(*t);
  (*t) = NULL;
}

int pllValidateNewick(pllNewickTree *t) {
  pllStack *head;
  pllNewickNodeInfo *item;
  int correct = 0;

  item = (pllNewickNodeInfo *)t->tree->item;
  if (item->rank != 2 && item->rank != 3)
    return (0);
  head = t->tree->next;
  while (head) {
    item = (pllNewickNodeInfo *)head->item;
    if (item->rank != 2 && item->rank != 0) {
      return (0);
    }
    head = head->next;
  }

  item = (pllNewickNodeInfo *)t->tree->item;

  if (item->rank == 2) {
    correct = (t->nodes == 2 * t->tips - 1);
    if (correct) {
      errno = PLL_NEWICK_ROOTED_TREE;
    } else {
      errno = PLL_NEWICK_BAD_STRUCTURE;
    }
    return (PLL_FALSE);
  }

  correct = ((t->nodes == 2 * t->tips - 2) && t->nodes != 4);
  if (correct)
    return (PLL_TRUE);

  errno = PLL_NEWICK_BAD_STRUCTURE;

  return (1);
}

int pllNewickUnroot(pllNewickTree *t) {
  pllStack *tmp;
  pllNewickNodeInfo *item;

  item = (pllNewickNodeInfo *)t->tree->item;
  if (item->rank == 2) {
    item->rank = 3;
    t->nodes--;
    item = (pllNewickNodeInfo *)t->tree->next->item;
    if (item->rank == 0) {
      tmp = t->tree->next->next;
      t->tree->next->next = t->tree->next->next->next;
    } else {
      tmp = t->tree->next;
      t->tree->next = t->tree->next->next;
    }
    item = (pllNewickNodeInfo *)tmp->item;
    rax_free(item->name);
    rax_free(tmp->item);
    rax_free(tmp);
  }

  return (pllValidateNewick(t));
}