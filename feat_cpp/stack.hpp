#pragma once
struct pllStack
{
    void *item;
    struct pllStack *next;
};

void pllStackClear(pllStack **stack);
void *pllStackPop(pllStack **head);
int pllStackPush(pllStack **head, void *item);
int pllStackSize(pllStack **stack);