#include <stdio.h>
#include "stack.hpp"
#include "mem_alloc.hpp"

int pllStackSize (pllStack ** stack)
{
  pllStack * top;
  int size = 0;
  top = *stack;

  while (top)
  {
    ++ size;
    top = top->next;
  }

  return (size);
}

int pllStackPush (pllStack ** head, void * item)
{
  pllStack *elem;

  elem = (pllStack *) rax_malloc (sizeof (pllStack));
  if (!elem) return (0);

  elem->item = item;
  elem->next = *head;
  *head     = elem;

  return (1);
}

void * pllStackPop (pllStack ** head)
{
  void * item;
  pllStack * tmp;
  if (!*head) return (NULL);

  tmp     = (*head);
  item    = (*head)->item;
  (*head) = (*head)->next;
  rax_free (tmp);

  return (item);
}

void pllStackClear (pllStack ** stack)
{
  while (*stack) pllStackPop (stack);
}
